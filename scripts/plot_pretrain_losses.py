#!/usr/bin/env python3
"""Plot pretraining loss curves from cramming output logs.

The script intentionally uses the raw pretrain logs as the source of truth,
because some short/failed runs never emit convergence CSVs.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


LOSS_RE = re.compile(
    r"Train loss (?P<loss>[-+0-9.eE]+) at step (?P<step>\d+) with lr (?P<lr>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"loaded with (?P<params>[0-9,]+) parameters")


@dataclass
class RunCurve:
    model: str
    run_id: str
    log_path: Path
    steps: list[int]
    losses: list[float]
    lrs: list[float]
    params: int | None

    @property
    def points(self) -> int:
        return len(self.steps)

    @property
    def final_step(self) -> int:
        return self.steps[-1]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    @property
    def min_loss(self) -> float:
        return min(self.losses)

    @property
    def label(self) -> str:
        if self.params:
            return f"{self.model} ({self.params / 1_000_000:.2f}M)"
        return self.model

    @property
    def run_label(self) -> str:
        return f"{self.model}/{self.run_id}"


def discover_logs(outputs_dir: Path) -> list[Path]:
    logs = []
    for path in outputs_dir.rglob("*.log"):
        parts = set(path.parts)
        if "pretrain" not in parts:
            continue
        if ".ipynb_checkpoints" in parts:
            continue
        logs.append(path)
    return sorted(logs)


def parse_log(path: Path, outputs_dir: Path) -> RunCurve | None:
    rel = path.relative_to(outputs_dir)
    model = rel.parts[0]
    try:
        pretrain_index = rel.parts.index("pretrain")
        run_id = "/".join(rel.parts[pretrain_index + 1 : -1])
    except ValueError:
        run_id = path.parent.name

    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    params: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if params is None:
                param_match = PARAM_RE.search(line)
                if param_match:
                    params = int(param_match.group("params").replace(",", ""))
            loss_match = LOSS_RE.search(line)
            if not loss_match:
                continue
            steps.append(int(loss_match.group("step")))
            losses.append(float(loss_match.group("loss").rstrip(".")))
            lrs.append(float(loss_match.group("lr").rstrip(".")))

    if not steps:
        return None

    ordered = sorted(zip(steps, losses, lrs), key=lambda row: row[0])
    steps = [row[0] for row in ordered]
    losses = [row[1] for row in ordered]
    lrs = [row[2] for row in ordered]
    return RunCurve(model, run_id, path, steps, losses, lrs, params)


def longest_curve_per_model(curves: list[RunCurve]) -> list[RunCurve]:
    best: dict[str, RunCurve] = {}
    for curve in curves:
        current = best.get(curve.model)
        if current is None:
            best[curve.model] = curve
            continue
        if (curve.final_step, curve.points, str(curve.log_path)) > (
            current.final_step,
            current.points,
            str(current.log_path),
        ):
            best[curve.model] = curve
    return sorted(best.values(), key=lambda curve: curve.final_step, reverse=True)


def write_summary(path: Path, curves: list[RunCurve]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "run_id",
                "points",
                "final_step",
                "final_loss",
                "min_loss",
                "params",
                "source",
            ],
        )
        writer.writeheader()
        for curve in curves:
            writer.writerow(
                {
                    "model": curve.model,
                    "run_id": curve.run_id,
                    "points": curve.points,
                    "final_step": curve.final_step,
                    "final_loss": f"{curve.final_loss:.6f}",
                    "min_loss": f"{curve.min_loss:.6f}",
                    "params": curve.params or "",
                    "source": str(curve.log_path),
                }
            )


def write_skipped(path: Path, logs: list[Path], curves: list[RunCurve]) -> None:
    parsed = {curve.log_path for curve in curves}
    skipped = [log for log in logs if log not in parsed]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "reason"])
        writer.writeheader()
        for log in skipped:
            writer.writerow({"source": str(log), "reason": "no Train loss rows"})


def write_missing_models(path: Path, outputs_dir: Path, curves: list[RunCurve]) -> None:
    plotted_models = {curve.model for curve in curves}
    output_models = sorted(item.name for item in outputs_dir.iterdir() if item.is_dir())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "reason"])
        writer.writeheader()
        for model in output_models:
            if model not in plotted_models:
                writer.writerow({"model": model, "reason": "no usable pretrain loss curve"})


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        total += value
        queue.append(value)
        if len(queue) > window:
            total -= queue.pop(0)
        smoothed.append(total / len(queue))
    return smoothed


def plot_curves(path_base: Path, curves: list[RunCurve], all_runs: bool, smooth: int) -> None:
    import matplotlib.pyplot as plt

    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig_width = 14
    fig_height = max(7, 0.55 * len(curves) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for curve in curves:
        label = curve.run_label if all_runs else curve.label
        losses = moving_average(curve.losses, smooth)
        ax.plot(curve.steps, losses, linewidth=1.7 if all_runs else 2.2, label=label)
        ax.scatter([curve.final_step], [losses[-1]], s=22)

    title = "All pretrain runs: loss vs step" if all_runs else "All models: pretrain loss vs step"
    subtitle = "raw log curves"
    if smooth > 1:
        subtitle += f", {smooth}-point moving average"
    ax.set_title(f"{title}\n{subtitle}", fontsize=15)
    ax.set_xlabel("Step")
    ax.set_ylabel("MLM train loss")
    ax.set_xlim(left=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--tables", type=Path, default=Path("tables"))
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    logs = discover_logs(args.outputs)
    curves = [curve for log in logs if (curve := parse_log(log, args.outputs))]
    if not curves:
        raise SystemExit(f"No pretrain loss curves found in {args.outputs}")

    model_curves = longest_curve_per_model(curves)
    write_summary(args.tables / "all_models_pretrain_loss_from_logs_summary.csv", model_curves)
    write_summary(args.tables / "all_runs_pretrain_loss_from_logs_summary.csv", curves)
    write_skipped(args.tables / "skipped_pretrain_logs_summary.csv", logs, curves)
    write_missing_models(args.tables / "missing_models_pretrain_loss_summary.csv", args.outputs, curves)
    plot_curves(args.tables / "all_models_pretrain_loss_vs_step", model_curves, all_runs=False, smooth=args.smooth)
    plot_curves(args.tables / "all_runs_pretrain_loss_vs_step", curves, all_runs=True, smooth=args.smooth)

    print(f"Parsed {len(curves)} pretrain runs from {len(logs)} logs.")
    print(f"Plotted {len(model_curves)} model curves.")


if __name__ == "__main__":
    main()
