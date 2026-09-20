#!/usr/bin/env bash
set -euo pipefail

# RR paper ablations with a fixed number of optimizer updates per run.
# train.steps is converted to microbatch iterations using the global batch,
# microbatch size, and number of local GPU processes.
#
# Usage:
#   bash scripts/rr_paper_fixed_updates.sh print-pretrain core
#   bash scripts/rr_paper_fixed_updates.sh pretrain core
#   bash scripts/rr_paper_fixed_updates.sh pretrain components
#   bash scripts/rr_paper_fixed_updates.sh eval core
#
# Repeating the same command resumes matching runs from intermediate_state.pth.
# Use the same prefix, global batch, microbatch size, and GPU count when resuming.
# Different microbatch/GPU settings get a distinct prefix by default.
#
# Examples:
#   TRAIN_MBS=128 bash scripts/rr_paper_fixed_updates.sh pretrain core
#   NPROC_PER_NODE=4 TRAIN_MBS=512 TRAIN_BATCH=2048 \
#     bash scripts/rr_paper_fixed_updates.sh pretrain core
#   FIXED_UPDATES=100000 bash scripts/rr_paper_fixed_updates.sh dryrun-pretrain core

ACTION="${1:-print-pretrain}"
GROUP="${2:-core}"

FIXED_UPDATES="${FIXED_UPDATES:-1000000}"
TRAIN_MBS="${TRAIN_MBS:-256}"
TRAIN_BATCH="${TRAIN_BATCH:-2048}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEED="${SEED:-1975620753}"
PREFIX="${PREFIX:-rr_fixed_${FIXED_UPDATES}_b${TRAIN_BATCH}_mbs${TRAIN_MBS}_g${NPROC_PER_NODE}_s${SEED}}"

export FIXED_UPDATES TRAIN_MBS TRAIN_BATCH NPROC_PER_NODE SEED PREFIX
export BUDGET=0
export AUTO_MICROBATCH=False
export RESUME_RUN_AFTER_PREEMPT=True
export SAVE_EVERY_NTH_STEP="${SAVE_EVERY_NTH_STEP:-100000}"

exec bash scripts/rr_paper_ablations.sh "$ACTION" "$GROUP"
