#!/usr/bin/env bash
set -euo pipefail

# Deep-supervision / ACT halt-step sweep for TRM h256.
#
# The defaults are meant to be comfortable on a 24GB GPU: TRAIN_MBS=16,
# TRAIN_BATCH=TRAIN_MBS, and compile_torch=False. If the first run has headroom,
# try TRAIN_MBS=32 TRAIN_BATCH=32.
#
# Usage:
#   bash scripts/trm_deep_supervision_sweep.sh print-pretrain
#   bash scripts/trm_deep_supervision_sweep.sh pretrain
#   bash scripts/trm_deep_supervision_sweep.sh dryrun-pretrain
#   bash scripts/trm_deep_supervision_sweep.sh print-eval
#   bash scripts/trm_deep_supervision_sweep.sh eval
#   bash scripts/trm_deep_supervision_sweep.sh dryrun-eval
#
# Useful overrides:
#   PRETRAIN_DS_STEPS="1 2 4 8 16" FINETUNE_DS_STEPS="1 2 4" bash scripts/trm_deep_supervision_sweep.sh print-eval
#   PREFIX=my_trm BUDGET=24 TRAIN_MBS=32 TRAIN_BATCH=32 bash scripts/trm_deep_supervision_sweep.sh pretrain
#   INFERENCE_STEPS=16 bash scripts/trm_deep_supervision_sweep.sh eval
#
# Note: paper TRM uses 8 heads at hidden_size=512. This h256 config keeps
# 8 heads and uses head_dim=32.

ACTION="${1:-print-pretrain}"

PREFIX="${PREFIX:-trm_h256_ds}"
SEED="${SEED:-1975620753}"
BUDGET="${BUDGET:-8}"
DATA="${DATA:-pile-readymade}"
TRAIN_CFG="${TRAIN_CFG:-rr-me-onecycle}"
ARCH_CFG="${ARCH_CFG:-trm-paper-h256}"
TRAIN_MBS="${TRAIN_MBS:-16}"
TRAIN_BATCH="${TRAIN_BATCH:-$TRAIN_MBS}"
EVAL_MBS="${EVAL_MBS:-8}"
EVAL_CFG="${EVAL_CFG:-GLUE_sane}"
EVAL_EPOCHS="${EVAL_EPOCHS:-4}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LR="${EVAL_LR:-8e-5}"
COMPILE_TORCH="${COMPILE_TORCH:-False}"
PRETRAIN_DS_STEPS="${PRETRAIN_DS_STEPS:-1 2 4 8 16}"
FINETUNE_DS_STEPS="${FINETUNE_DS_STEPS:-1 2 4}"
INFERENCE_STEPS="${INFERENCE_STEPS:-match-finetune}"
HALT_EXPLORATION_PROB="${HALT_EXPLORATION_PROB:-0.1}"

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

comment() {
  if [[ "$EXECUTE" != "true" ]]; then
    printf '# %s\n' "$*"
  fi
}

pretrain_name() {
  local pt_ds="$1"
  echo "${PREFIX}_ptds${pt_ds}"
}

pretrain_cmd() {
  local pt_ds="$1"
  local name
  shift
  name="$(pretrain_name "$pt_ds")"

  run_cmd \
    python pretrain.py \
    name="$name" \
    seed="$SEED" \
    data="$DATA" \
    train="$TRAIN_CFG" \
    arch="$ARCH_CFG" \
    budget="$BUDGET" \
    train.batch_size="$TRAIN_BATCH" \
    dryrun="$DRYRUN" \
    impl.microbatch_size="$TRAIN_MBS" \
    impl.compile_torch="$COMPILE_TORCH" \
    arch.deep_supervision_steps="$pt_ds" \
    arch.inference_steps="$pt_ds" \
    arch.halt_max_steps="$pt_ds" \
    arch.act_training=True \
    arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" \
    "wandb.tags=[trm,h256,deep-supervision,pretrain]" \
    "$@"
}

eval_cmd() {
  local pt_ds="$1"
  local ft_ds="$2"
  local name
  local eval_steps
  shift 2
  name="$(pretrain_name "$pt_ds")"

  if [[ "$INFERENCE_STEPS" == "match-finetune" ]]; then
    eval_steps="$ft_ds"
  else
    eval_steps="$INFERENCE_STEPS"
  fi

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
    eval.arch_modifications.deep_supervision_steps="$ft_ds" \
    eval.arch_modifications.inference_steps="$eval_steps" \
    eval.arch_modifications.halt_max_steps="$ft_ds" \
    eval.arch_modifications.act_training=False \
    "wandb.tags=[trm,h256,deep-supervision,glue,ptds${pt_ds},ftds${ft_ds},infs${eval_steps}]" \
    "$@"
}

if [[ "$PHASE" == "pretrain" ]]; then
  comment "pretrain halt_max_steps: ${PRETRAIN_DS_STEPS}"
  comment "mirroring halt_max_steps to deep_supervision_steps for config compatibility"
  comment "train batch/microbatch: ${TRAIN_BATCH}/${TRAIN_MBS}"
  comment "halt_exploration_prob: ${HALT_EXPLORATION_PROB}"
  for pt_ds in $PRETRAIN_DS_STEPS; do
    pretrain_cmd "$pt_ds"
  done
else
  comment "pretrain deep_supervision_steps: ${PRETRAIN_DS_STEPS}"
  comment "fine-tune deep_supervision_steps: ${FINETUNE_DS_STEPS}"
  comment "inference_steps: ${INFERENCE_STEPS}"
  for pt_ds in $PRETRAIN_DS_STEPS; do
    for ft_ds in $FINETUNE_DS_STEPS; do
      eval_cmd "$pt_ds" "$ft_ds"
    done
  done
fi
