#!/usr/bin/env bash
set -euo pipefail

# Epoch-based paper ablations for Recursive Refiner.
#
# This mirrors rr_paper_ablations.sh, but pretraining duration is derived from
# dataset epochs instead of wall-clock budget:
#
#   train.steps = floor(PILE_ROWS / impl.microbatch_size) * TRAIN_EPOCHS
#
# The budget remains only as a safety ceiling for pretrain.py.
#
# Usage:
#   bash scripts/rr_paper_one_epoch_pretrain.sh print-pretrain core
#   bash scripts/rr_paper_one_epoch_pretrain.sh pretrain core
#   bash scripts/rr_paper_one_epoch_pretrain.sh print-eval core
#   bash scripts/rr_paper_one_epoch_pretrain.sh eval core
#   TRAIN_EPOCHS=3 bash scripts/rr_paper_one_epoch_pretrain.sh print-pretrain all
#   NUM_GPUS=4 bash scripts/rr_paper_one_epoch_pretrain.sh pretrain all
#   GPU_IDS=0,2,4 bash scripts/rr_paper_one_epoch_pretrain.sh pretrain all
#
# NUM_GPUS model runs are executed concurrently, one model per GPU. GPU_IDS
# selects explicit devices and implies NUM_GPUS when NUM_GPUS is not set.
#
# Groups:
#   core        - smallest defensible table around the successful tiny8x model
#   baselines   - BERT/CrammedBERT/ALBERT comparisons at matched dimensions/depths
#   components  - RR component ablations: embedding rank, cycles, depth, FFN, norm
#   sizes       - RR width scaling sweep
#   all         - core + baselines + components + sizes

ACTION="${1:-print-pretrain}"
GROUP="${2:-core}"

PREFIX="${PREFIX:-rr_paper_${TRAIN_EPOCHS:-${EPOCHS:-1}}ep}"
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
REQUESTED_NUM_GPUS="${NUM_GPUS:-${N_GPUS:-${3:-}}}"
GPU_IDS_CSV="${GPU_IDS:-}"

PHASE=""
EXECUTE="false"
DRYRUN="False"
NUM_GPUS=""
PRINT_GPU_INDEX=0
SCHEDULER_TMP_DIR=""
GPU_FIFO=""
GPU_FD=""
declare -a GPU_LIST=()
declare -a JOB_PIDS=()
declare -a JOB_NAMES=()
declare -A SEEN_GPU_IDS=()

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

if ! [[ "$TRAIN_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_EPOCHS must be a positive integer, got: $TRAIN_EPOCHS" >&2
  exit 2
fi

if [[ -n "$GPU_IDS_CSV" ]]; then
  if [[ "$GPU_IDS_CSV" == ,* || "$GPU_IDS_CSV" == *, || "$GPU_IDS_CSV" == *,,* ]]; then
    echo "GPU_IDS must not contain empty entries: $GPU_IDS_CSV" >&2
    exit 2
  fi

  IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS_CSV"
  for gpu_id in "${GPU_LIST[@]}"; do
    if [[ -z "$gpu_id" || "$gpu_id" =~ [[:space:]] ]]; then
      echo "GPU_IDS must be a comma-separated list without empty or whitespace-containing entries: $GPU_IDS_CSV" >&2
      exit 2
    fi
    if [[ -n "${SEEN_GPU_IDS[$gpu_id]:-}" ]]; then
      echo "GPU_IDS contains a duplicate device: $gpu_id" >&2
      exit 2
    fi
    SEEN_GPU_IDS["$gpu_id"]=1
  done

  NUM_GPUS="${#GPU_LIST[@]}"
  if [[ -n "$REQUESTED_NUM_GPUS" ]]; then
    if ! [[ "$REQUESTED_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
      echo "NUM_GPUS must be a positive integer, got: $REQUESTED_NUM_GPUS" >&2
      exit 2
    fi
    if [[ "$REQUESTED_NUM_GPUS" != "$NUM_GPUS" ]]; then
      echo "NUM_GPUS ($REQUESTED_NUM_GPUS) must match the number of entries in GPU_IDS ($NUM_GPUS)" >&2
      exit 2
    fi
  fi
else
  NUM_GPUS="${REQUESTED_NUM_GPUS:-1}"
  if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
    exit 2
  fi
  for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
    GPU_LIST+=("$gpu_id")
  done
fi

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 2
fi

cleanup_scheduler() {
  if [[ -n "$GPU_FD" ]]; then
    exec {GPU_FD}>&-
    GPU_FD=""
  fi
  if [[ -n "$GPU_FIFO" ]]; then
    rm -f "$GPU_FIFO"
  fi
  if [[ -n "$SCHEDULER_TMP_DIR" ]]; then
    rmdir "$SCHEDULER_TMP_DIR" 2>/dev/null || true
  fi
}

terminate_jobs() {
  local exit_code="${1:-130}"
  local pid

  trap - INT TERM
  for pid in "${JOB_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  cleanup_scheduler
  exit "$exit_code"
}

init_scheduler() {
  local gpu_id

  if [[ "$EXECUTE" != "true" ]]; then
    return 0
  fi

  SCHEDULER_TMP_DIR="$(mktemp -d)"
  GPU_FIFO="$SCHEDULER_TMP_DIR/gpus"
  mkfifo "$GPU_FIFO"
  exec {GPU_FD}<>"$GPU_FIFO"
  rm -f "$GPU_FIFO"
  GPU_FIFO=""

  for gpu_id in "${GPU_LIST[@]}"; do
    printf '%s\n' "$gpu_id" >&"$GPU_FD"
  done

  trap cleanup_scheduler EXIT
  trap 'terminate_jobs 130' INT TERM
}

command_name() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      name=*)
        printf '%s\n' "${arg#name=}"
        return
        ;;
    esac
  done
  printf '%s\n' "$1"
}

run_cmd() {
  local gpu_id
  local job_name

  if [[ "$EXECUTE" != "true" ]]; then
    gpu_id="${GPU_LIST[$PRINT_GPU_INDEX]}"
    PRINT_GPU_INDEX=$(( (PRINT_GPU_INDEX + 1) % NUM_GPUS ))
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
    printf '%q ' "$@"
    printf '\n'
    return
  fi

  IFS= read -r gpu_id <&"$GPU_FD"
  job_name="$(command_name "$@")"
  printf '[gpu %s] starting %s\n' "$gpu_id" "$job_name"

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu_id" "$@"
    status=$?
    printf '%s\n' "$gpu_id" >&"$GPU_FD"
    exit "$status"
  ) &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("$job_name")
}

wait_for_jobs() {
  local index
  local status
  local failed=0

  for index in "${!JOB_PIDS[@]}"; do
    if wait "${JOB_PIDS[$index]}"; then
      printf '[done] %s\n' "${JOB_NAMES[$index]}"
    else
      status=$?
      printf '[failed:%s] %s\n' "$status" "${JOB_NAMES[$index]}" >&2
      failed=1
    fi
  done

  return "$failed"
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

  comment "name=${name} rows=${PILE_ROWS} epochs=${TRAIN_EPOCHS} model_gpus=1 microbatch=${mbs} steps=${steps}"

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
    "wandb.tags=[rr-paper,epoch-pretrain]" \
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
  # Core uses full-rank embeddings; embedding-factorization sweeps live in components.
  rr tiny8x_h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  rr tiny8x_h256_l2_c2x3_ef1 256 4 2 2 3 1 4.0

  # Same dimensions as tiny8x, but no nested RR state/cycles.
  albert_shared h256_eff16_e256 256 4 16 1024 256

  # Same hidden width and physical depth.
  hfbert h256_l2 256 4 2 1024
  crammed h256_l2 256 4 2 1024

  # Smaller full-embedding BERT baseline.
  hfbert h128_l2_param_match 128 2 2 512
}

group_baselines() {
  group_core

  # Match RR's effective Transformer-block applications: 2 layers * 2 hi * (3 lo + 1 hi) = 16.
  hfbert h256_l16_effective_depth 256 4 16 1024 impl.microbatch_size=128
  crammed h256_l16_effective_depth 256 4 16 1024 impl.microbatch_size=128

  # Shared-weight controls at the same effective depth with ALBERT factorized embeddings.
  albert_shared h256_eff8_e64 256 4 8 1024 64
}

group_components() {
  # Low-rank/tied embedding ablation.
  rr h256_l2_c2x3_ef1 256 4 2 2 3 1 4.0
  rr h256_l2_c2x3_ef2 256 4 2 2 3 2 4.0
  rr h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
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
  rr h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  rr h512_l2_c2x3_ef4 512 8 2 2 3 4 4.0 impl.microbatch_size=128
  rr h768_l2_c2x3_ef4 768 12 2 2 3 4 4.0 impl.microbatch_size=128
  rr h1024_l2_c2x3_ef4 1024 16 2 2 3 4 4.0 impl.microbatch_size=64
}

init_scheduler

case "$GROUP" in
  core) group_core ;;
  baselines) group_baselines ;;
  components) group_components ;;
  sizes) group_sizes ;;
  all)
    group_baselines
    group_components
    group_sizes
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

if [[ "$EXECUTE" == "true" ]]; then
  if ! wait_for_jobs; then
    exit 1
  fi
fi
