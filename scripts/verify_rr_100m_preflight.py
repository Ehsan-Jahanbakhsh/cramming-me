#!/usr/bin/env python3
"""Preflight checks for the 100M Recursive Refiner run.

Run this on the actual training VM before spending a long budget:

    python scripts/verify_rr_100m_preflight.py --device cuda

It checks two things that are hard to verify in this Windows workspace:
1. The configured 100M-class RR model constructs and has the expected size.
2. SDPA attention agrees with the manual attention fallback on small tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    max_abs: float
    mean_abs: float
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip-model", action="store_true", help="Skip full 100M model construction")
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-3)
    return parser.parse_args()


def resolve_torch_dtype(torch, dtype_name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def count_params(model) -> int:
    return sum(param.numel() for param in model.parameters())


def make_100m_config(RecursiveRefinerConfig):
    return RecursiveRefinerConfig(
        vocab_size=32768,
        max_position_embeddings=512,
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=3,
        expansion=8.0,
        hi_cycles=2,
        lo_cycles=3,
        embed_factor=4,
        pre_norm=True,
        rms_eps=1e-5,
        rope_theta=10000.0,
        is_causal=False,
        use_sdpa=True,
    )


def compare_attention(torch, SelfAttention, device: str, dtype, causal: bool, with_mask: bool, atol: float, rtol: float) -> CheckResult:
    torch.manual_seed(1337)
    dim = 64
    heads = 4
    seq_len = 17
    batch_size = 3

    sdpa = SelfAttention(dim=dim, num_heads=heads, is_causal=causal, use_sdpa=True).to(device=device, dtype=dtype)
    manual = SelfAttention(dim=dim, num_heads=heads, is_causal=causal, use_sdpa=False).to(device=device, dtype=dtype)
    manual.load_state_dict(sdpa.state_dict())
    sdpa.eval()
    manual.eval()

    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    attention_mask = None
    if with_mask:
        attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        attention_mask[0, -3:] = 0
        attention_mask[1, -5:] = 0

    with torch.inference_mode():
        y_sdpa = sdpa(x, attention_mask=attention_mask)
        y_manual = manual(x, attention_mask=attention_mask)

    diff = (y_sdpa.float() - y_manual.float()).abs()
    name = f"attention causal={causal} mask={with_mask}"
    return CheckResult(
        name=name,
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        passed=bool(torch.allclose(y_sdpa.float(), y_manual.float(), atol=atol, rtol=rtol)),
    )


def main() -> None:
    args = parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    dtype = resolve_torch_dtype(torch, args.dtype)

    from cramming.architectures.recursive_refiner_hf import (
        RecursiveRefinerConfig,
        RecursiveRefinerForMaskedLM,
        SelfAttention,
    )

    print(f"torch={torch.__version__} device={args.device} dtype={args.dtype}")
    print(f"sdpa_available={hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")

    if not args.skip_model:
        cfg = make_100m_config(RecursiveRefinerConfig)
        model = RecursiveRefinerForMaskedLM(cfg)
        params = count_params(model)
        print(f"rr_100m_params={params:,} ({params / 1_000_000:.2f}M)")
        if not (95_000_000 <= params <= 100_000_000):
            raise SystemExit("Unexpected RR 100M parameter count.")

    checks = [
        compare_attention(torch, SelfAttention, args.device, dtype, causal=False, with_mask=False, atol=args.atol, rtol=args.rtol),
        compare_attention(torch, SelfAttention, args.device, dtype, causal=False, with_mask=True, atol=args.atol, rtol=args.rtol),
        compare_attention(torch, SelfAttention, args.device, dtype, causal=True, with_mask=False, atol=args.atol, rtol=args.rtol),
        compare_attention(torch, SelfAttention, args.device, dtype, causal=True, with_mask=True, atol=args.atol, rtol=args.rtol),
    ]

    failed = False
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}: max_abs={result.max_abs:.6g} mean_abs={result.mean_abs:.6g}")
        failed = failed or not result.passed

    if failed:
        raise SystemExit("SDPA parity check failed. Run final smoke with arch.use_sdpa=false or inspect kernels/tolerances.")


if __name__ == "__main__":
    main()
