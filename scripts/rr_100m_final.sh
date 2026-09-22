#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/rr_100m_final.sh print
#   bash scripts/rr_100m_final.sh preflight
#   bash scripts/rr_100m_final.sh dryrun
#   bash scripts/rr_100m_final.sh smoke20
#   bash scripts/rr_100m_final.sh smoke30
#   bash scripts/rr_100m_final.sh final
#   bash scripts/rr_100m_final.sh print-eval
#   bash scripts/rr_100m_final.sh eval
#   bash scripts/rr_100m_final.sh summarize-eval
#   bash scripts/rr_100m_final.sh wide-final
#
# Environment overrides:
#   SEED=1337 BUDGET=72 TRAIN_MBS=128 COMPILE_TORCH=True bash scripts/rr_100m_final.sh final

ACTION="${1:-print}"
SEED="${SEED:-1337}"
BUDGET="${BUDGET:-72}"
SMOKE_BUDGET="${SMOKE_BUDGET:-2}"
TRAIN_MBS="${TRAIN_MBS:-128}"
WIDE_TRAIN_MBS="${WIDE_TRAIN_MBS:-96}"
COMPILE_TORCH="${COMPILE_TORCH:-True}"
DATA="${DATA:-pile-readymade}"
DRYRUN="${DRYRUN:-False}"
PREFLIGHT_DEVICE="${PREFLIGHT_DEVICE:-cuda}"
PREFLIGHT_DTYPE="${PREFLIGHT_DTYPE:-float32}"
EVAL_NAME="${EVAL_NAME:-rr_100m_h1024_l3_exp8_mlm30}"
EVAL_CFG="${EVAL_CFG:-GLUE_sane}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-latest}"
EVAL_EPOCHS="${EVAL_EPOCHS:-5}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LR="${EVAL_LR:-8e-5}"
EVAL_MBS="${EVAL_MBS:-16}"
EVAL_COMPILE_TORCH="${EVAL_COMPILE_TORCH:-False}"
SUMMARY_OUTPUTS="${SUMMARY_OUTPUTS:-outputs}"
SUMMARY_TABLES="${SUMMARY_TABLES:-tables}"

run_or_print() {
  if [[ "$ACTION" == "print" || "$ACTION" == "print-eval" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

base_cmd() {
  local name="$1"
  local budget="$2"
  local mlm="$3"
  local mbs="$4"
  shift 4

  run_or_print \
    python pretrain.py \
    name="$name" \
    seed="$SEED" \
    data="$DATA" \
    train=rr-modern-wsd \
    arch=recursive-refiner-100m \
    budget="$budget" \
    dryrun="$DRYRUN" \
    impl.microbatch_size="$mbs" \
    impl.compile_torch="$COMPILE_TORCH" \
    train.objective.mlm_probability="$mlm" \
    "wandb.tags=[rr-final,100m,mlm${mlm/./}]" \
    "$@"
}

eval_cmd() {
  local name="$1"
  shift

  run_or_print \
    python eval.py \
    name="$name" \
    seed="$SEED" \
    eval="$EVAL_CFG" \
    eval.checkpoint="$EVAL_CHECKPOINT" \
    eval.epochs="$EVAL_EPOCHS" \
    eval.batch_size="$EVAL_BATCH" \
    eval.optim.lr="$EVAL_LR" \
    eval.epoch_selection=best \
    eval.selection_metric=target \
    dryrun="$DRYRUN" \
    impl.microbatch_size="$EVAL_MBS" \
    impl.shuffle_in_dataloader=True \
    impl.compile_torch="$EVAL_COMPILE_TORCH" \
    "wandb.tags=[rr-final,100m,eval]" \
    "$@"
}

case "$ACTION" in
  print)
    base_cmd rr_100m_h1024_l3_exp8_mlm30 "$BUDGET" 0.30 "$TRAIN_MBS"
    ;;
  preflight)
    python scripts/verify_rr_100m_preflight.py --device "$PREFLIGHT_DEVICE" --dtype "$PREFLIGHT_DTYPE"
    ;;
  dryrun)
    DRYRUN=True
    base_cmd rr_100m_dryrun 1 0.30 "$TRAIN_MBS"
    ;;
  smoke20)
    base_cmd rr_100m_smoke_mlm20 "$SMOKE_BUDGET" 0.20 "$TRAIN_MBS"
    ;;
  smoke30)
    base_cmd rr_100m_smoke_mlm30 "$SMOKE_BUDGET" 0.30 "$TRAIN_MBS"
    ;;
  final)
    base_cmd rr_100m_h1024_l3_exp8_mlm30 "$BUDGET" 0.30 "$TRAIN_MBS"
    ;;
  print-eval)
    eval_cmd "$EVAL_NAME"
    ;;
  eval)
    eval_cmd "$EVAL_NAME"
    ;;
  summarize-eval)
    python scripts/extract_best_eval_from_logs.py --outputs "$SUMMARY_OUTPUTS" --tables "$SUMMARY_TABLES"
    ;;
  wide-final)
    base_cmd rr_100m_h1280_l2_exp8_mlm30 "$BUDGET" 0.30 "$WIDE_TRAIN_MBS" \
      arch=recursive-refiner-large \
      arch.hidden_size=1280 \
      arch.num_attention_heads=20 \
      arch.num_hidden_layers=2 \
      arch.expansion=8.0 \
      arch.hi_cycles=2 \
      arch.lo_cycles=3 \
      arch.embed_factor=4
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
