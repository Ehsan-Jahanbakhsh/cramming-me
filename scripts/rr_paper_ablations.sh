#!/usr/bin/env bash
set -euo pipefail

# Paper ablations for Recursive Refiner.
#
# Usage:
#   bash scripts/rr_paper_ablations.sh print-pretrain core
#   bash scripts/rr_paper_ablations.sh pretrain core
#   bash scripts/rr_paper_ablations.sh print-eval core
#   bash scripts/rr_paper_ablations.sh eval core
#   bash scripts/rr_paper_ablations.sh dryrun-pretrain core
#
# Groups:
#   core        - smallest defensible table around the successful tiny8x model
#   baselines   - additional effective-depth BERT/ALBERT controls after core
#   components  - RR component variants after the core embedding anchors
#   sizes       - RR width variants after the h256 core anchor
#   all         - core + baselines + components + sizes
#
# Resume the same named runs with:
#   RESUME_RUN_AFTER_PREEMPT=True bash scripts/rr_paper_ablations.sh pretrain core
# Disable automatic GPU sizing with AUTO_MICROBATCH=False.
# Cap automatic sizing with AUTO_MICROBATCH_MAX_SIZE=256.
# TRAIN_MBS is used when automatic sizing is disabled.

ACTION="${1:-print-pretrain}"
GROUP="${2:-core}"

PREFIX="${PREFIX:-rr_paper_v2}"
SEED="${SEED:-1975620753}"
BUDGET="${BUDGET:-8}"
TRAIN_MBS="${TRAIN_MBS:-256}"
TRAIN_BATCH="${TRAIN_BATCH:-2048}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
FIXED_UPDATES="${FIXED_UPDATES:-}"
EVAL_MBS="${EVAL_MBS:-16}"
EVAL_CFG="${EVAL_CFG:-GLUE}"
EVAL_EPOCHS="${EVAL_EPOCHS:-4}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LR="${EVAL_LR:-8e-5}"
COMPILE_TORCH="${COMPILE_TORCH:-True}"
AUTO_MICROBATCH="${AUTO_MICROBATCH:-True}"
AUTO_MICROBATCH_MAX_SIZE="${AUTO_MICROBATCH_MAX_SIZE:-null}"
RESUME_RUN_AFTER_PREEMPT="${RESUME_RUN_AFTER_PREEMPT:-False}"
SAVE_EVERY_NTH_STEP="${SAVE_EVERY_NTH_STEP:-100000}"

if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got: $NPROC_PER_NODE" >&2
  exit 2
fi

PHASE=""
EXECUTE="false"
DRYRUN="False"

case "$ACTION" in
  print-pretrain) PHASE="pretrain" ;;
  print-eval) PHASE="eval" ;;
  pretrain) PHASE="pretrain"; EXECUTE="true" ;;
  eval) PHASE="eval"; EXECUTE="true" ;;
  dryrun-pretrain) PHASE="pretrain"; EXECUTE="true"; DRYRUN="True" ;;
  dryrun-eval) PHASE="eval"; EXECUTE="true"; DRYRUN="True" ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac

run_cmd() {
  if [[ "$EXECUTE" == "true" ]]; then
    "$@"
  else
    printf '%q ' "$@"
    printf '\n'
  fi
}

pretrain_cmd() {
  local name="$1"
  shift

  local mbs="$TRAIN_MBS"
  local batch="$TRAIN_BATCH"
  local steps=""
  local arg
  local extra_args=()
  local run_overrides=()
  local launcher=(python pretrain.py)

  for arg in "$@"; do
    case "$arg" in
      impl.microbatch_size=*) mbs="${arg#impl.microbatch_size=}" ;;
      train.batch_size=*) batch="${arg#train.batch_size=}" ;;
      train.steps=*|train.scheduler=*|train.batch_size_ramp=*|train.warmup_steps=*|train.cooldown_steps=*)
        if [[ -n "$FIXED_UPDATES" ]]; then
          echo "FIXED_UPDATES controls $arg; remove this per-run override." >&2
          exit 2
        fi
        extra_args+=("$arg")
        ;;
      *) extra_args+=("$arg") ;;
    esac
  done

  if (( NPROC_PER_NODE > 1 )); then
    launcher=(torchrun --nproc_per_node="$NPROC_PER_NODE" --standalone pretrain.py)
  fi

  if [[ -n "$FIXED_UPDATES" ]]; then
    if ! [[ "$FIXED_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
      echo "FIXED_UPDATES must be a positive integer, got: $FIXED_UPDATES" >&2
      exit 2
    fi
    if ! [[ "$mbs" =~ ^[1-9][0-9]*$ && "$batch" =~ ^[1-9][0-9]*$ ]]; then
      echo "TRAIN_MBS and TRAIN_BATCH must be positive integers for fixed-update runs." >&2
      exit 2
    fi
    case "$AUTO_MICROBATCH" in
      True|TRUE|true|1|yes|YES)
        echo "Fixed-update runs require AUTO_MICROBATCH=False so the script can derive train.steps." >&2
        exit 2
        ;;
    esac
    local effective_mbs=$((mbs * NPROC_PER_NODE))
    if (( effective_mbs > batch || batch % effective_mbs != 0 )); then
      echo "Global batch ($batch) must be divisible by microbatch_size * NPROC_PER_NODE ($effective_mbs)." >&2
      exit 2
    fi

    # train.steps counts microbatch iterations in this trainer. Convert the
    # requested optimizer updates using the fixed global batch and accumulation.
    steps=$((FIXED_UPDATES * batch / effective_mbs))
    run_overrides+=(
      train.steps="$steps"
      train.scheduler=one-cycle
      train.batch_size_ramp=0
      train.warmup_steps=0
      train.cooldown_steps=0
    )
  fi

  run_cmd \
    "${launcher[@]}" \
    name="$name" \
    seed="$SEED" \
    data=pile-readymade \
    train=rr-me-onecycle \
    budget="$BUDGET" \
    dryrun="$DRYRUN" \
    train.batch_size="$batch" \
    impl.microbatch_size="$mbs" \
    impl.auto_microbatch="$AUTO_MICROBATCH" \
    impl.auto_microbatch_max_size="$AUTO_MICROBATCH_MAX_SIZE" \
    impl.save_intermediate_checkpoints=True \
    impl.save_every_nth_step="$SAVE_EVERY_NTH_STEP" \
    impl.resume_run_after_preempt="$RESUME_RUN_AFTER_PREEMPT" \
    impl.compile_torch="$COMPILE_TORCH" \
    "wandb.tags=[rr-paper,pretrain]" \
    "${run_overrides[@]}" \
    "${extra_args[@]}"
}

eval_cmd() {
  local name="$1"
  shift
  run_cmd \
    python eval.py \
    name="$name" \
    seed="$SEED" \
    eval="$EVAL_CFG" \
    eval.checkpoint=latest \
    eval.epochs="$EVAL_EPOCHS" \
    eval.batch_size="$EVAL_BATCH" \
    eval.optim.lr="$EVAL_LR" \
    dryrun="$DRYRUN" \
    impl.microbatch_size="$EVAL_MBS" \
    impl.shuffle_in_dataloader=True \
    impl.compile_torch=False \
    "wandb.tags=[rr-paper,eval]" \
    "$@"
}

emit() {
  local name="$1"
  shift
  if [[ "$PHASE" == "pretrain" ]]; then
    pretrain_cmd "$name" "$@"
  else
    eval_cmd "$name"
  fi
}

rr() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local hi="$5"
  local lo="$6"
  local embed_factor="$7"
  local expansion="$8"
  shift 8
  emit "${PREFIX}_rr_${suffix}" \
    arch=recursive-refiner-tiny \
    arch.hidden_size="$hidden" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$layers" \
    arch.hi_cycles="$hi" \
    arch.lo_cycles="$lo" \
    arch.embed_factor="$embed_factor" \
    arch.expansion="$expansion" \
    "$@"
}

rr_flat() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local cycles="$5"
  local embed_factor="$6"
  local expansion="$7"
  shift 7
  emit "${PREFIX}_rrflat_${suffix}" \
    arch=recursive-refiner-tiny \
    arch.recurrence_mode=flat \
    arch.hidden_size="$hidden" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$layers" \
    arch.flat_cycles="$cycles" \
    arch.hi_cycles=1 \
    arch.lo_cycles=1 \
    arch.grad_last_cycle_only=False \
    arch.embed_factor="$embed_factor" \
    arch.expansion="$expansion" \
    "$@"
}

rr_flat_untied() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local cycles="$5"
  local embed_factor="$6"
  local expansion="$7"
  shift 7
  emit "${PREFIX}_rrflatuntied_${suffix}" \
    arch=recursive-refiner-tiny \
    arch.recurrence_mode=flat_untied \
    arch.hidden_size="$hidden" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$layers" \
    arch.flat_cycles="$cycles" \
    arch.hi_cycles=1 \
    arch.lo_cycles=1 \
    arch.grad_last_cycle_only=False \
    arch.embed_factor="$embed_factor" \
    arch.expansion="$expansion" \
    "$@"
}

hfbert() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local intermediate="$5"
  shift 5
  emit "${PREFIX}_hfbert_${suffix}" \
    arch=hf-bert-tiny \
    arch.hidden_size="$hidden" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$layers" \
    arch.intermediate_size="$intermediate" \
    "$@"
}

crammed() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local intermediate="$5"
  shift 5
  emit "${PREFIX}_crammed_${suffix}" \
    arch=crammed-bert \
    arch.hidden_size="$hidden" \
    arch.num_transformer_layers="$layers" \
    arch.intermed_size="$intermediate" \
    arch.attention.num_attention_heads="$heads" \
    arch.embedding.embedding_dim="$hidden" \
    arch.classification_head.head_dim="$hidden" \
    "$@"
}

albert_shared() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local effective_layers="$4"
  local intermediate="$5"
  local embedding="$6"
  shift 6
  emit "${PREFIX}_albert_${suffix}" \
    arch=hf-albert-shared \
    arch.hidden_size="$hidden" \
    arch.embedding_size="$embedding" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$effective_layers" \
    arch.intermediate_size="$intermediate" \
    arch.num_hidden_groups=1 \
    arch.inner_group_num=1 \
    "$@"
}

group_core() {
  # Rank-1 and rank-4 embedding anchors; remaining ranks live in components.
  rr tiny8x_h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  rr tiny8x_h256_l2_c2x3_ef1 256 4 2 2 3 1 4.0

  # Same dimensions as tiny8x, but no nested RR state/cycles.
  albert_shared h256_eff16_e256 256 4 16 1024 256

  # Exact RR block/embedding control with one latent and flat recurrence.
  # hi=2, lo=3 makes 8 shared-stack passes (16 physical block applications).
  rr_flat h256_l2_flat8_ef4 256 4 2 8 4 4.0

  # Untied control: same 16 applications and injection rule, with 8x the RR block parameters.
  # Embeddings are re-injected once before each pair of blocks, as in rr_flat.
  rr_flat_untied h256_l16_flat8_ef4 256 4 2 8 4 4.0

  # Same hidden width and physical depth.
  hfbert h256_l2 256 4 2 1024
  crammed h256_l2 256 4 2 1024

  # Smaller full-embedding BERT baseline.
  hfbert h128_l2_param_match 128 2 2 512
}

group_baselines() {
  # Match RR's effective Transformer-block applications: 2 layers * 2 hi * (3 lo + 1 hi) = 16.
  hfbert h256_l16_effective_depth 256 4 16 1024 impl.microbatch_size=128
  crammed h256_l16_effective_depth 256 4 16 1024 impl.microbatch_size=128

  # Shared-weight controls at the same effective depth with ALBERT factorized embeddings.
  albert_shared h256_eff8_e64 256 4 8 1024 64
}

group_components() {
  # The ef1 and ef4 anchors are in core; run core before this group.
  rr h256_l2_c2x3_ef2 256 4 2 2 3 2 4.0
  rr h256_l2_c2x3_ef8 256 4 2 2 3 8 4.0

  # Recurrence schedule at roughly fixed physical parameters.
  rr h256_l2_c1x1_ef4 256 4 2 1 1 4 4.0
  rr h256_l2_c1x3_ef4 256 4 2 1 3 4 4.0
  rr h256_l2_c2x1_ef4 256 4 2 2 1 4 4.0
  rr h256_l2_c4x1_ef4 256 4 2 4 1 4 4.0
  rr h256_l2_c1x7_ef4 256 4 2 1 7 4 4.0
  rr h256_l2_c3x2_ef4 256 4 2 3 2 4 4.0

  # Physical block count while preserving the successful 2x3 schedule.
  rr h256_l1_c2x3_ef4 256 4 1 2 3 4 4.0
  rr h256_l4_c2x3_ef4 256 4 4 2 3 4 4.0 impl.microbatch_size=128

  # FFN and normalization choices.
  rr h256_l2_c2x3_ef4_exp2 256 4 2 2 3 4 2.0
  rr h256_l2_c2x3_ef4_exp6 256 4 2 2 3 4 6.0
  rr h256_l2_c2x3_ef4_postnorm 256 4 2 2 3 4 4.0 arch.pre_norm=False
}

group_sizes() {
  rr h128_l2_c2x3_ef4 128 2 2 2 3 4 4.0
  # h256 is the core anchor.
  rr h512_l2_c2x3_ef4 512 8 2 2 3 4 4.0 impl.microbatch_size=128
  rr h768_l2_c2x3_ef4 768 12 2 2 3 4 4.0 impl.microbatch_size=128
  rr h1024_l2_c2x3_ef4 1024 16 2 2 3 4 4.0 impl.microbatch_size=64
}

case "$GROUP" in
  core) group_core ;;
  baselines) group_baselines ;;
  components) group_components ;;
  sizes) group_sizes ;;
  all)
    group_core
    group_baselines
    group_components
    group_sizes
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
