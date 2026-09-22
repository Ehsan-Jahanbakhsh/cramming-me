param(
  [string]$Action = "print"
)

# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 print
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 preflight
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 dryrun
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 smoke20
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 smoke30
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 final
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 print-eval
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 eval
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 summarize-eval
#   powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 wide-final

$Seed = if ($env:SEED) { $env:SEED } else { "1337" }
$Budget = if ($env:BUDGET) { $env:BUDGET } else { "72" }
$SmokeBudget = if ($env:SMOKE_BUDGET) { $env:SMOKE_BUDGET } else { "2" }
$TrainMbs = if ($env:TRAIN_MBS) { $env:TRAIN_MBS } else { "128" }
$WideTrainMbs = if ($env:WIDE_TRAIN_MBS) { $env:WIDE_TRAIN_MBS } else { "96" }
$CompileTorch = if ($env:COMPILE_TORCH) { $env:COMPILE_TORCH } else { "True" }
$Data = if ($env:DATA) { $env:DATA } else { "pile-readymade" }
$Dryrun = if ($env:DRYRUN) { $env:DRYRUN } else { "False" }
$PreflightDevice = if ($env:PREFLIGHT_DEVICE) { $env:PREFLIGHT_DEVICE } else { "cuda" }
$PreflightDtype = if ($env:PREFLIGHT_DTYPE) { $env:PREFLIGHT_DTYPE } else { "float32" }
$EvalName = if ($env:EVAL_NAME) { $env:EVAL_NAME } else { "rr_100m_h1024_l3_exp8_mlm30" }
$EvalCfg = if ($env:EVAL_CFG) { $env:EVAL_CFG } else { "GLUE_sane" }
$EvalCheckpoint = if ($env:EVAL_CHECKPOINT) { $env:EVAL_CHECKPOINT } else { "latest" }
$EvalEpochs = if ($env:EVAL_EPOCHS) { $env:EVAL_EPOCHS } else { "5" }
$EvalBatch = if ($env:EVAL_BATCH) { $env:EVAL_BATCH } else { "16" }
$EvalLr = if ($env:EVAL_LR) { $env:EVAL_LR } else { "8e-5" }
$EvalMbs = if ($env:EVAL_MBS) { $env:EVAL_MBS } else { "16" }
$EvalCompileTorch = if ($env:EVAL_COMPILE_TORCH) { $env:EVAL_COMPILE_TORCH } else { "False" }
$SummaryOutputs = if ($env:SUMMARY_OUTPUTS) { $env:SUMMARY_OUTPUTS } else { "outputs" }
$SummaryTables = if ($env:SUMMARY_TABLES) { $env:SUMMARY_TABLES } else { "tables" }

function Format-Arg {
  param([string]$Arg)
  if ($Arg -match '[\s\[\],]') {
    return '"' + $Arg.Replace('"', '\"') + '"'
  }
  return $Arg
}

function Invoke-OrPrint {
  param([string[]]$PythonArgs)
  if ($Action -eq "print" -or $Action -eq "print-eval") {
    $rendered = @("python") + ($PythonArgs | ForEach-Object { Format-Arg $_ })
    Write-Output ($rendered -join " ")
  } else {
    & python @PythonArgs
  }
}

function Invoke-Base {
  param(
    [string]$Name,
    [string]$RunBudget,
    [string]$Mlm,
    [string]$Mbs,
    [string[]]$ExtraArgs = @()
  )

  $args = @(
    "pretrain.py",
    "name=$Name",
    "seed=$Seed",
    "data=$Data",
    "train=rr-modern-wsd",
    "arch=recursive-refiner-100m",
    "budget=$RunBudget",
    "dryrun=$Dryrun",
    "impl.microbatch_size=$Mbs",
    "impl.compile_torch=$CompileTorch",
    "train.objective.mlm_probability=$Mlm",
    "wandb.tags=[rr-final,100m,mlm$($Mlm.Replace('.', ''))]"
  ) + $ExtraArgs

  Invoke-OrPrint $args
}

function Invoke-Eval {
  param([string]$Name)

  $args = @(
    "eval.py",
    "name=$Name",
    "seed=$Seed",
    "eval=$EvalCfg",
    "eval.checkpoint=$EvalCheckpoint",
    "eval.epochs=$EvalEpochs",
    "eval.batch_size=$EvalBatch",
    "eval.optim.lr=$EvalLr",
    "eval.epoch_selection=best",
    "eval.selection_metric=target",
    "dryrun=$Dryrun",
    "impl.microbatch_size=$EvalMbs",
    "impl.shuffle_in_dataloader=True",
    "impl.compile_torch=$EvalCompileTorch",
    "wandb.tags=[rr-final,100m,eval]"
  )

  Invoke-OrPrint $args
}

switch ($Action) {
  "print" {
    Invoke-Base "rr_100m_h1024_l3_exp8_mlm30" $Budget "0.30" $TrainMbs
  }
  "preflight" {
    & python "scripts/verify_rr_100m_preflight.py" "--device" $PreflightDevice "--dtype" $PreflightDtype
  }
  "dryrun" {
    $script:Dryrun = "True"
    Invoke-Base "rr_100m_dryrun" "1" "0.30" $TrainMbs
  }
  "smoke20" {
    Invoke-Base "rr_100m_smoke_mlm20" $SmokeBudget "0.20" $TrainMbs
  }
  "smoke30" {
    Invoke-Base "rr_100m_smoke_mlm30" $SmokeBudget "0.30" $TrainMbs
  }
  "final" {
    Invoke-Base "rr_100m_h1024_l3_exp8_mlm30" $Budget "0.30" $TrainMbs
  }
  "print-eval" {
    Invoke-Eval $EvalName
  }
  "eval" {
    Invoke-Eval $EvalName
  }
  "summarize-eval" {
    & python "scripts/extract_best_eval_from_logs.py" "--outputs" $SummaryOutputs "--tables" $SummaryTables
  }
  "wide-final" {
    Invoke-Base "rr_100m_h1280_l2_exp8_mlm30" $Budget "0.30" $WideTrainMbs @(
      "arch=recursive-refiner-large",
      "arch.hidden_size=1280",
      "arch.num_attention_heads=20",
      "arch.num_hidden_layers=2",
      "arch.expansion=8.0",
      "arch.hi_cycles=2",
      "arch.lo_cycles=3",
      "arch.embed_factor=4"
    )
  }
  default {
    Write-Error "Unknown action: $Action"
    exit 2
  }
}
