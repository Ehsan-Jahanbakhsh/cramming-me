#!/usr/bin/env bash
set -euo pipefail

# Four-GPU TRM/RR refinement ablations matched to rr_single_high_tiny8x, budget defaults to 8h.
# Usage: bash scripts/trm_refinement_ablatio_4gpu.sh
# Print commands without running: RUN=echo DRYRUN=True bash scripts/trm_refinement_ablatio_4gpu.sh
# Skip eval after pretraining: DO_EVAL=False bash scripts/trm_refinement_ablatio_4gpu.sh
# On Windows, if plain bash opens WSL, call your Git/MinGW bash executable directly.

RUN="${RUN:-}"
PREFIX="${PREFIX:-rr_single_high_cmp_b8_4gpu}"
SEED="${SEED:-1975620753}"
BUDGET="${BUDGET:-8}"
DATA="${DATA:-pile-readymade}"
DATA_STREAMING="${DATA_STREAMING:-False}"
TRAIN_CFG="${TRAIN_CFG:-rr-me-onecycle}"
RR_ARCH_CFG="${RR_ARCH_CFG:-recursive-refiner-tiny}"
TRM_ARCH_CFG="${TRM_ARCH_CFG:-trm}"
HIDDEN="${HIDDEN:-128}"
HEADS="${HEADS:-2}"
LAYERS="${LAYERS:-2}"
LO="${LO:-3}"
EMBED_FACTOR="${EMBED_FACTOR:-4}"
EXPANSION="${EXPANSION:-4.0}"
HALT_LOSS="${HALT_LOSS:-0.0}"
HALT_EXPLORATION_PROB="${HALT_EXPLORATION_PROB:-0.0}"
TRAIN_MBS="${TRAIN_MBS:-256}"
TRAIN_BATCH="${TRAIN_BATCH:-2048}"
EVAL_MBS="${EVAL_MBS:-16}"
EVAL_CFG="${EVAL_CFG:-GLUE_sane}"
EVAL_EPOCHS="${EVAL_EPOCHS:-4}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LR="${EVAL_LR:-8e-5}"
COMPILE_TORCH="${COMPILE_TORCH:-True}"
DRYRUN="${DRYRUN:-False}"
DO_EVAL="${DO_EVAL:-True}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PRETRAIN_EXTRA="${PRETRAIN_EXTRA:-}"
EVAL_EXTRA="${EVAL_EXTRA:-}"

wait_wave() {
  local label="$1"
  local failed=0
  wait "$p0" || failed=1
  wait "$p1" || failed=1
  wait "$p2" || failed=1
  wait "$p3" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    echo "$label failed" >&2
    exit 1
  fi
}

echo "Pretrain wave 1: GPUs ${GPU0},${GPU1},${GPU2},${GPU3}"
# GPU0: RR anchor at the intended 8h budget, same shape as your 24h rr_single_high_tiny8x command.
$RUN env CUDA_VISIBLE_DEVICES="$GPU0" python pretrain.py name="${PREFIX}_rr_h2_l${LO}_b${BUDGET}" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$RR_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=2 arch.lo_cycles="$LO" arch.grad_last_cycle_only=False arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" "wandb.tags=[rr-single-high,pretrain,tiny8x,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p0=$!
# GPU1: TRM single-step control, no carried deep supervision, full gradients through inner cycles.
$RUN env CUDA_VISIBLE_DEVICES="$GPU1" python pretrain.py name="${PREFIX}_trm_h2_l${LO}_ds1_gradfull" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$TRM_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=2 arch.lo_cycles="$LO" arch.deep_supervision_steps=1 arch.inference_steps=1 arch.halt_max_steps=1 arch.grad_last_cycle_only=False arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" arch.q_halt_loss_weight="$HALT_LOSS" arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" "wandb.tags=[trm-refine,pretrain,baseline,gradfull,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p1=$!
# GPU2: TRM single-step with last-cycle-only gradients, isolating gradient truncation.
$RUN env CUDA_VISIBLE_DEVICES="$GPU2" python pretrain.py name="${PREFIX}_trm_h2_l${LO}_ds1_gradlast" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$TRM_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=2 arch.lo_cycles="$LO" arch.deep_supervision_steps=1 arch.inference_steps=1 arch.halt_max_steps=1 arch.grad_last_cycle_only=True arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" arch.q_halt_loss_weight="$HALT_LOSS" arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" "wandb.tags=[trm-refine,pretrain,baseline,gradlast,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p2=$!
# GPU3: TRM carried refinement with full gradients.
$RUN env CUDA_VISIBLE_DEVICES="$GPU3" python pretrain.py name="${PREFIX}_trm_h2_l${LO}_ds2_gradfull" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$TRM_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=2 arch.lo_cycles="$LO" arch.deep_supervision_steps=2 arch.inference_steps=2 arch.halt_max_steps=2 arch.grad_last_cycle_only=False arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" arch.q_halt_loss_weight="$HALT_LOSS" arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" "wandb.tags=[trm-refine,pretrain,deep2,gradfull,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p3=$!
wait_wave "pretrain wave 1"

echo "Pretrain wave 2: GPUs ${GPU0},${GPU1},${GPU2},${GPU3}"
# GPU0: TRM carried refinement with last-cycle-only gradients, main TRM-like candidate.
$RUN env CUDA_VISIBLE_DEVICES="$GPU0" python pretrain.py name="${PREFIX}_trm_h2_l${LO}_ds2_gradlast" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$TRM_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=2 arch.lo_cycles="$LO" arch.deep_supervision_steps=2 arch.inference_steps=2 arch.halt_max_steps=2 arch.grad_last_cycle_only=True arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" arch.q_halt_loss_weight="$HALT_LOSS" arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" "wandb.tags=[trm-refine,pretrain,deep2,gradlast,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p0=$!
# GPU1: Inner-depth alternative, spend compute on one long inner refinement instead of carried revision.
$RUN env CUDA_VISIBLE_DEVICES="$GPU1" python pretrain.py name="${PREFIX}_trm_h4_l${LO}_ds1_gradlast" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$TRM_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=4 arch.lo_cycles="$LO" arch.deep_supervision_steps=1 arch.inference_steps=1 arch.halt_max_steps=1 arch.grad_last_cycle_only=True arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" arch.q_halt_loss_weight="$HALT_LOSS" arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" "wandb.tags=[trm-refine,pretrain,inner-depth,gradlast,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p1=$!
# GPU2: Outer-depth alternative, four carried revisions with shallow inner refinement.
$RUN env CUDA_VISIBLE_DEVICES="$GPU2" python pretrain.py name="${PREFIX}_trm_h1_l${LO}_ds4_gradlast" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$TRM_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=1 arch.lo_cycles="$LO" arch.deep_supervision_steps=4 arch.inference_steps=4 arch.halt_max_steps=4 arch.grad_last_cycle_only=True arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" arch.q_halt_loss_weight="$HALT_LOSS" arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" "wandb.tags=[trm-refine,pretrain,outer-depth,gradlast,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p2=$!
# GPU3: Memory-stretch probe, spend saved activation memory on more high cycles.
$RUN env CUDA_VISIBLE_DEVICES="$GPU3" python pretrain.py name="${PREFIX}_trm_h6_l${LO}_ds1_gradlast" seed="$SEED" data="$DATA" train="$TRAIN_CFG" arch="$TRM_ARCH_CFG" budget="$BUDGET" dryrun="$DRYRUN" data.streaming="$DATA_STREAMING" train.batch_size="$TRAIN_BATCH" impl.microbatch_size="$TRAIN_MBS" impl.compile_torch="$COMPILE_TORCH" arch.hidden_size="$HIDDEN" arch.num_attention_heads="$HEADS" arch.num_hidden_layers="$LAYERS" arch.hi_cycles=6 arch.lo_cycles="$LO" arch.deep_supervision_steps=1 arch.inference_steps=1 arch.halt_max_steps=1 arch.grad_last_cycle_only=True arch.embed_factor="$EMBED_FACTOR" arch.expansion="$EXPANSION" arch.q_halt_loss_weight="$HALT_LOSS" arch.halt_exploration_prob="$HALT_EXPLORATION_PROB" "wandb.tags=[trm-refine,pretrain,memory-stretch,gradlast,budget${BUDGET},4gpu]" $PRETRAIN_EXTRA &
p3=$!
wait_wave "pretrain wave 2"

if [[ "$DO_EVAL" == "True" || "$DO_EVAL" == "true" || "$DO_EVAL" == "1" ]]; then
  echo "Eval wave 1: GPUs ${GPU0},${GPU1},${GPU2},${GPU3}"
  # GPU0: Eval RR anchor.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU0" python eval.py name="${PREFIX}_rr_h2_l${LO}_b${BUDGET}" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False "wandb.tags=[rr-single-high,eval,tiny8x,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p0=$!
  # GPU1: Eval TRM single-step full-gradient control.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU1" python eval.py name="${PREFIX}_trm_h2_l${LO}_ds1_gradfull" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False eval.arch_modifications.deep_supervision_steps=1 eval.arch_modifications.inference_steps=1 eval.arch_modifications.halt_max_steps=1 eval.arch_modifications.grad_last_cycle_only=False eval.arch_modifications.q_halt_loss_weight="$HALT_LOSS" "wandb.tags=[trm-refine,eval,baseline,gradfull,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p1=$!
  # GPU2: Eval TRM single-step last-cycle-only control.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU2" python eval.py name="${PREFIX}_trm_h2_l${LO}_ds1_gradlast" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False eval.arch_modifications.deep_supervision_steps=1 eval.arch_modifications.inference_steps=1 eval.arch_modifications.halt_max_steps=1 eval.arch_modifications.grad_last_cycle_only=True eval.arch_modifications.q_halt_loss_weight="$HALT_LOSS" "wandb.tags=[trm-refine,eval,baseline,gradlast,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p2=$!
  # GPU3: Eval TRM carried refinement with full gradients.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU3" python eval.py name="${PREFIX}_trm_h2_l${LO}_ds2_gradfull" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False eval.arch_modifications.deep_supervision_steps=2 eval.arch_modifications.inference_steps=2 eval.arch_modifications.halt_max_steps=2 eval.arch_modifications.grad_last_cycle_only=False eval.arch_modifications.q_halt_loss_weight="$HALT_LOSS" "wandb.tags=[trm-refine,eval,deep2,gradfull,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p3=$!
  wait_wave "eval wave 1"

  echo "Eval wave 2: GPUs ${GPU0},${GPU1},${GPU2},${GPU3}"
  # GPU0: Eval TRM carried refinement with last-cycle-only gradients.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU0" python eval.py name="${PREFIX}_trm_h2_l${LO}_ds2_gradlast" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False eval.arch_modifications.deep_supervision_steps=2 eval.arch_modifications.inference_steps=2 eval.arch_modifications.halt_max_steps=2 eval.arch_modifications.grad_last_cycle_only=True eval.arch_modifications.q_halt_loss_weight="$HALT_LOSS" "wandb.tags=[trm-refine,eval,deep2,gradlast,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p0=$!
  # GPU1: Eval inner-depth alternative.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU1" python eval.py name="${PREFIX}_trm_h4_l${LO}_ds1_gradlast" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False eval.arch_modifications.deep_supervision_steps=1 eval.arch_modifications.inference_steps=1 eval.arch_modifications.halt_max_steps=1 eval.arch_modifications.grad_last_cycle_only=True eval.arch_modifications.q_halt_loss_weight="$HALT_LOSS" "wandb.tags=[trm-refine,eval,inner-depth,gradlast,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p1=$!
  # GPU2: Eval outer-depth alternative.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU2" python eval.py name="${PREFIX}_trm_h1_l${LO}_ds4_gradlast" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False eval.arch_modifications.deep_supervision_steps=4 eval.arch_modifications.inference_steps=4 eval.arch_modifications.halt_max_steps=4 eval.arch_modifications.grad_last_cycle_only=True eval.arch_modifications.q_halt_loss_weight="$HALT_LOSS" "wandb.tags=[trm-refine,eval,outer-depth,gradlast,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p2=$!
  # GPU3: Eval memory-stretch probe.
  $RUN env CUDA_VISIBLE_DEVICES="$GPU3" python eval.py name="${PREFIX}_trm_h6_l${LO}_ds1_gradlast" seed="$SEED" eval="$EVAL_CFG" eval.checkpoint=latest eval.epochs="$EVAL_EPOCHS" eval.batch_size="$EVAL_BATCH" eval.optim.lr="$EVAL_LR" dryrun="$DRYRUN" impl.microbatch_size="$EVAL_MBS" impl.shuffle_in_dataloader=True impl.compile_torch=False eval.arch_modifications.deep_supervision_steps=1 eval.arch_modifications.inference_steps=1 eval.arch_modifications.halt_max_steps=1 eval.arch_modifications.grad_last_cycle_only=True eval.arch_modifications.q_halt_loss_weight="$HALT_LOSS" "wandb.tags=[trm-refine,eval,memory-stretch,gradlast,budget${BUDGET},4gpu]" $EVAL_EXTRA &
  p3=$!
  wait_wave "eval wave 2"
fi
