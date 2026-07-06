"""Script to evaluate a pretrained model."""

import torch
import hydra


import time
import datetime
import logging
from collections import defaultdict

import cramming
import evaluate


log = logging.getLogger(__name__)


def main_downstream_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    tasks = cramming.prepare_task_dataloaders(tokenizer, cfg.eval, cfg.impl)

    metrics = dict()
    stats = defaultdict(list)
    # Start the clocks now:
    for task_name, task in tasks.items():
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        # Prepare model for finetuning:
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        model_engine.load_checkpoint(cfg_arch, model_file)

        try:
            assert task_name != "record"
            metric = evaluate.load(task["details"]["collection"], task_name, cache_dir=cfg.impl.path)
        except (FileNotFoundError, AssertionError):  # no specific metric downloadable from evaluate, construct directly
            targets = [evaluate.load(metric_name, cache_dir=cfg.impl.path) for metric_name in task["details"]["target_metrics"]]
            metric = evaluate.CombinedEvaluations(targets)
        # Launch training
        model_engine.train(cfg.eval.eval_in_train_mode)
        loss_vals = []
        epoch_selection = cfg.eval.get("epoch_selection", "last")
        selection_mode = cfg.eval.get("selection_mode", "max")
        if epoch_selection not in ["last", "best"]:
            raise ValueError(f"Invalid eval.epoch_selection={epoch_selection}. Choose 'last' or 'best'.")
        if epoch_selection == "best" and cfg.eval.evaluation_set == "test":
            log.warning("eval.epoch_selection=best is selecting on the test split. Use a validation split for paper comparisons.")

        best_score = None
        best_epoch = None
        best_metrics = None
        best_model_state = None
        for epoch in range(cfg.eval.epochs):
            train_time = time.time()

            for step, batch in enumerate(task["trainloader"]):
                # Heavy lifting is moved to engines
                device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
                loss = model_engine.step(device_batch)
                loss_vals.append(loss.detach())
                if cfg.dryrun:
                    break

            metrics[task_name] = validate(model_engine, task["validloader"], metric, setup, cfg)
            selection_score = _selection_score(metrics[task_name], task, cfg) if epoch_selection == "best" else float("nan")
            stats[f"{task_name}_epoch"] += [epoch]
            stats[f"{task_name}_loss"] += [loss.item()]

            stats[f"{task_name}_avg_loss"] += [torch.stack(loss_vals).mean().item()]  # Smoothed loss
            stats[f"{task_name}_selection_score"] += [selection_score]
            loss_vals = []
            current_lr = model_engine.optimizer.param_groups[0]["lr"]

            log_msg = f"Train loss {loss.item():2.4f} at step {step} with lr {current_lr:.5f}. "
            log_msg += f"[Avg: {stats[f'{task_name}_avg_loss'][-1]:2.4f}] after epoch {epoch}."

            stats[f"{task_name}_train_time"] += [(time.time() - train_time)]
            estimated_train_finish = str(datetime.timedelta(seconds=stats[f"{task_name}_train_time"][-1] * cfg.eval.epochs))
            tokens_per_second = (step + 1) * cfg.eval.max_seq_length * cfg.impl.microbatch_size / stats[f"{task_name}_train_time"][-1]
            log_msg += (
                f" Perf: {stats[f'{task_name}_train_time'][-1]/60:2.4f}min per epoch ({tokens_per_second:.0f}t/s). "
                f"Estimated Total Train: {estimated_train_finish}."
            )

            for name, metric_val in metrics[task_name].items():
                stats[f"{task_name}_{name}"] += [metric_val]
            if epoch_selection == "best" and _is_better(selection_score, best_score, selection_mode):
                best_score = selection_score
                best_epoch = epoch
                best_metrics = dict(metrics[task_name])
                best_model_state = _copy_model_state_to_cpu(model_engine)
                log_msg += f" New best {selection_score:2.4f}."
            log.info(log_msg)
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in metrics[task_name].items()])
            log.info(f"Validation metric is {msg_metrics} after epoch {epoch}.")
            cramming.utils.wandb_log(stats, cfg)

            if cfg.dryrun:
                break
        if epoch_selection == "best":
            metrics[task_name] = best_metrics
            _restore_model_state(model_engine, best_model_state)
            stats[f"{task_name}_selected_epoch"] += [best_epoch]
            stats[f"{task_name}_selected_score"] += [best_score]
            for name, metric_val in best_metrics.items():
                stats[f"{task_name}_selected_{name}"] += [metric_val]
            log.info(f"Selected epoch {best_epoch} for task {task_name} with validation score {best_score:2.4f}.")
            cramming.utils.wandb_log({k: v for k, v in stats.items() if k.startswith(f"{task_name}_selected_")}, cfg)
        # Launch extra testing if extra validation set exists (as with MNLI-mismatched):
        if task["extra_validloader"] is not None:
            extra_eval_metric = validate(model_engine, task["extra_validloader"], metric, setup, cfg)
            # metrics[task_name + "extra"] = extra_eval_metric
            metrics[task_name].update({f"{k}_extra": v for k, v in extra_eval_metric.items()})
            for name, metric_val in extra_eval_metric.items():
                stats[f"{task_name}_{name}_extra"] += [metric_val]
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in extra_eval_metric.items()])
            log.info(f"Extra validation metric is {msg_metrics} after finetuning.")
            cramming.utils.wandb_log({f"{task_name}_{k}_extra": [v] for k, v in extra_eval_metric.items()}, cfg)

    # Check average metric over all tasks:
    target_metrics = []
    for task_name, task in tasks.items():
        target_metric_names = task["details"]["target_metrics"]
        for metric_name in target_metric_names:
            target_metrics.append(metrics[task_name][metric_name])
    metrics[f"{cfg.eval.name}_amean"] = torch.as_tensor(target_metrics).mean().item()
    metrics[f"{cfg.eval.name}_hmean"] = torch.as_tensor(target_metrics).pow(-1).mean().pow(-1).item()
    log.info(f"Overall average metric on evaluation {cfg.eval.name} is {metrics[f'{cfg.eval.name}_amean']:.2f}.")
    cramming.utils.wandb_log(
        {f"{cfg.eval.name}_amean": [metrics[f"{cfg.eval.name}_amean"]], f"{cfg.eval.name}_hmean": [metrics[f"{cfg.eval.name}_hmean"]]},
        cfg,
    )

    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.save_summary("downstream", cfg, stats, time.time() - local_time, setup)
    return metrics  # will be dumped into yaml


@torch.no_grad()
def validate(model_engine, validloader, metric, setup, cfg):
    """Evaluate on validation set."""
    model_engine.eval()
    for step, batch in enumerate(validloader):
        device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
        _, predictions = model_engine.forward_inference(**device_batch)

        if getattr(metric, "config_name", "") != "multirc":
            metric.add_batch(predictions=predictions, references=device_batch["labels"])
        else:  # uuuuuughhhhh, whhyyy multirc
            pred_indices = range(step * predictions.shape[0], (step + 1) * predictions.shape[0])
            packages = [dict(idx=validloader.index_lookup[pred_indices[i]], prediction=p) for i, p in enumerate(predictions.cpu())]
            metric.add_batch(predictions=packages, references=batch["labels"])

        if cfg.dryrun and step > 1:
            break

    try:
        eval_metric = metric.compute()
    except ValueError:  # pearson corr computation will raise errors if metric values are NaN
        log.info("Value Error in metrics computation, maybe non-finite values in prediction. Returning backup score.")
        eval_metric = metric.compute(predictions=[0, 1], references=[1, 0])  # spoof terrible result if metric computation fails
    model_engine.train(cfg.eval.eval_in_train_mode)
    return {k: float(v) for k, v in eval_metric.items()}  # force float returns


def _selection_score(metrics, task, cfg):
    """Compute the scalar used to select the downstream fine-tuning epoch."""
    selection_metric = cfg.eval.get("selection_metric", "target")
    if selection_metric == "target":
        metric_names = [name for name in task["details"]["target_metrics"] if name in metrics]
        if len(metric_names) == 0:
            metric_names = list(metrics.keys())
    else:
        if selection_metric not in metrics:
            raise ValueError(f"eval.selection_metric={selection_metric} not found in validation metrics {list(metrics)}.")
        metric_names = [selection_metric]
    return torch.as_tensor([metrics[name] for name in metric_names]).float().mean().item()


def _is_better(score, best_score, mode):
    if mode == "max":
        return best_score is None or score > best_score
    if mode == "min":
        return best_score is None or score < best_score
    raise ValueError(f"Invalid eval.selection_mode={mode}. Choose 'max' or 'min'.")


def _copy_model_state_to_cpu(model_engine):
    model = getattr(model_engine, "model", model_engine)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _restore_model_state(model_engine, state_dict):
    model = getattr(model_engine, "model", model_engine)
    model.load_state_dict(state_dict)
    if hasattr(model_engine, "setup"):
        model.to(**model_engine.setup)


@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")


if __name__ == "__main__":
    launch()
