#!/usr/bin/env bash
set -euo pipefail

# Early-exit Recursive Refiner experiments and matched ALBERT controls.
#
# Usage:
#   bash scripts/rr_early_exit_experiments.sh print-pretrain core
#   bash scripts/rr_early_exit_experiments.sh dryrun-pretrain core
#   bash scripts/rr_early_exit_experiments.sh pretrain core
#   bash scripts/rr_early_exit_experiments.sh print-eval core
#   bash scripts/rr_early_exit_experiments.sh eval core
#   bash scripts/rr_early_exit_experiments.sh check core
#
# Groups:
#   core   - fixed RR, RR early exit, RR early exit + feedback, fixed ALBERT, ALBERT early exit
#   sweep  - halt-threshold sweep for RR/ALBERT early exit
#   all    - core + sweep

ACTION="${1:-print-pretrain}"
GROUP="${2:-core}"

PREFIX="${PREFIX:-rr_ee_${TRAIN_EPOCHS:-${EPOCHS:-1}}ep}"
SEED="${SEED:-1975620753}"
PILE_ROWS="${PILE_ROWS:-85000000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-${EPOCHS:-1}}"
SAFETY_BUDGET="${SAFETY_BUDGET:-9999}"
TRAIN_MBS="${TRAIN_MBS:-256}"
TRAIN_BATCH="${TRAIN_BATCH:-2048}"
TRAIN_SCHEDULER="${TRAIN_SCHEDULER:-one-cycle}"
TRAIN_BATCH_RAMP="${TRAIN_BATCH_RAMP:-0}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-0}"
TRAIN_COOLDOWN_STEPS="${TRAIN_COOLDOWN_STEPS:-0}"
EVAL_MBS="${EVAL_MBS:-16}"
EVAL_CFG="${EVAL_CFG:-GLUE}"
EVAL_EPOCHS="${EVAL_EPOCHS:-4}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LR="${EVAL_LR:-8e-5}"
COMPILE_TORCH="${COMPILE_TORCH:-True}"
BASE_DIR="${BASE_DIR:-outputs}"

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
  check) PHASE="check"; EXECUTE="true" ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac

if ! [[ "$TRAIN_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_EPOCHS must be a positive integer, got: $TRAIN_EPOCHS" >&2
  exit 2
fi

run_cmd() {
  if [[ "$EXECUTE" != "true" ]]; then
    printf '%q ' "$@"
    printf '\n'
    return
  fi
  "$@"
}

comment() {
  if [[ "$EXECUTE" != "true" ]]; then
    printf '# %s\n' "$*"
  fi
}

microbatch_from_args() {
  local mbs="$TRAIN_MBS"
  local arg
  for arg in "$@"; do
    case "$arg" in
      impl.microbatch_size=*) mbs="${arg#impl.microbatch_size=}" ;;
    esac
  done
  echo "$mbs"
}

steps_for_epochs() {
  local mbs="$1"
  local steps_per_epoch=$(( PILE_ROWS / mbs ))
  if (( steps_per_epoch < 1 )); then
    steps_per_epoch=1
  fi
  echo $(( steps_per_epoch * TRAIN_EPOCHS ))
}

pretrain_cmd() {
  local name="$1"
  shift
  local mbs
  local steps
  local arg
  local extra_args=()

  mbs="$(microbatch_from_args "$@")"
  steps="$(steps_for_epochs "$mbs")"
  for arg in "$@"; do
    case "$arg" in
      impl.microbatch_size=*) ;;
      *) extra_args+=("$arg") ;;
    esac
  done

  comment "name=${name} rows=${PILE_ROWS} epochs=${TRAIN_EPOCHS} microbatch=${mbs} steps=${steps}"
  run_cmd \
    python pretrain.py \
    name="$name" \
    seed="$SEED" \
    data=pile-readymade \
    train=rr-me-onecycle \
    budget="$SAFETY_BUDGET" \
    dryrun="$DRYRUN" \
    train.steps="$steps" \
    train.scheduler="$TRAIN_SCHEDULER" \
    train.batch_size="$TRAIN_BATCH" \
    train.batch_size_ramp="$TRAIN_BATCH_RAMP" \
    train.warmup_steps="$TRAIN_WARMUP_STEPS" \
    train.cooldown_steps="$TRAIN_COOLDOWN_STEPS" \
    impl.microbatch_size="$mbs" \
    impl.compile_torch="$COMPILE_TORCH" \
    "wandb.tags=[rr-early-exit,pretrain]" \
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
    "wandb.tags=[rr-early-exit,eval]" \
    "$@"
}

check_cmd() {
  local name="$1"
  shift
  case "$name" in
    *_rr_ee*|*_albert_ee*)
      run_cmd python scripts/check_early_exit_stats.py "$name" --base-dir "$BASE_DIR" "$@"
      ;;
    *)
      comment "skip check for fixed-depth run ${name}"
      ;;
  esac
}

emit() {
  local name="$1"
  shift
  case "$PHASE" in
    pretrain) pretrain_cmd "$name" "$@" ;;
    eval) eval_cmd "$name" ;;
    check) check_cmd "$name" ;;
  esac
}

rr_fixed() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local hi="$5"
  local lo="$6"
  local embed_factor="$7"
  local expansion="$8"
  shift 8
  emit "${PREFIX}_rr_fixed_${suffix}" \
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

rr_ee() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local max_depth="$5"
  local lo="$6"
  local embed_factor="$7"
  local expansion="$8"
  local feedback="$9"
  local threshold="${10}"
  shift 10
  emit "${PREFIX}_rr_ee_${suffix}" \
    arch=recursive-refiner-ee-tiny \
    arch.hidden_size="$hidden" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$layers" \
    arch.hi_cycles="$max_depth" \
    arch.lo_cycles="$lo" \
    arch.embed_factor="$embed_factor" \
    arch.expansion="$expansion" \
    arch.early_exit.max_depth="$max_depth" \
    arch.early_exit.halt_threshold="$threshold" \
    arch.prediction_feedback.enabled="$feedback" \
    "$@"
}

albert_fixed() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local effective_layers="$4"
  local intermediate="$5"
  local embedding="$6"
  shift 6
  emit "${PREFIX}_albert_fixed_${suffix}" \
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

albert_ee() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local max_depth="$4"
  local exit_interval="$5"
  local intermediate="$6"
  local embedding="$7"
  local threshold="$8"
  shift 8
  emit "${PREFIX}_albert_ee_${suffix}" \
    arch=hf-albert-ee-shared \
    arch.hidden_size="$hidden" \
    arch.embedding_size="$embedding" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$(( max_depth * exit_interval ))" \
    arch.exit_interval="$exit_interval" \
    arch.early_exit.max_depth="$max_depth" \
    arch.early_exit.halt_threshold="$threshold" \
    arch.intermediate_size="$intermediate" \
    arch.num_hidden_groups=1 \
    arch.inner_group_num=1 \
    "$@"
}

group_core() {
  # RR max effective depth = high_depth * physical_layers * (lo_cycles + 1) = 2 * 2 * 4 = 16.
  rr_fixed h256_l2_d2_lo3_ef4 256 4 2 2 3 4 4.0
  rr_ee h256_l2_d2_lo3_ef4_t050 256 4 2 2 3 4 4.0 false 0.50
  rr_ee h256_l2_d2_lo3_ef4_fb_t050 256 4 2 2 3 4 4.0 true 0.50

  albert_fixed h256_eff16_e256 256 4 16 1024 256
  albert_ee h256_d2_i8_e256_t050 256 4 2 8 1024 256 0.50
}

group_sweep() {
  rr_ee h256_l2_d2_lo3_ef4_t035 256 4 2 2 3 4 4.0 false 0.35
  rr_ee h256_l2_d2_lo3_ef4_t050 256 4 2 2 3 4 4.0 false 0.50
  rr_ee h256_l2_d2_lo3_ef4_t065 256 4 2 2 3 4 4.0 false 0.65
  albert_ee h256_d2_i8_e256_t035 256 4 2 8 1024 256 0.35
  albert_ee h256_d2_i8_e256_t050 256 4 2 8 1024 256 0.50
  albert_ee h256_d2_i8_e256_t065 256 4 2 8 1024 256 0.65
}

case "$GROUP" in
  core) group_core ;;
  sweep) group_sweep ;;
  all)
    group_core
    group_sweep
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
