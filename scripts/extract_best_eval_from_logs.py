#!/usr/bin/env python3
"""Extract best validation metrics from cramming downstream eval logs.

This is intentionally log-based. Some Cramming summary YAML files contain the
last epoch, while paper comparisons usually want best validation epoch per task.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


TARGET_METRICS = {
    "cola": ["matthews_correlation"],
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "mnli": ["accuracy", "accuracy_extra"],
    "mrpc": ["accuracy", "f1"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "stsb": ["pearson", "spearmanr"],
}
GLUE_TASKS = ["cola", "rte", "sst2", "mnli", "mrpc", "qnli", "qqp", "stsb"]

FINETUNE_RE = re.compile(r"Finetuning task (?P<task>[A-Za-z0-9_]+) with")
VALID_RE = re.compile(r"Validation metric is (?P<body>.+?) after epoch (?P<epoch>\d+)")
EXTRA_RE = re.compile(r"Extra validation metric is (?P<body>.+?) after finetuning")
SELECTED_RE = re.compile(r"Selected epoch (?P<epoch>\d+) for task (?P<task>[A-Za-z0-9_]+)")
METRIC_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+): (?P<value>[-+0-9.eE]+)")
OVERALL_RE = re.compile(r"Overall average metric on evaluation (?P<name>[A-Za-z0-9_]+) is (?P<value>[-+0-9.eE]+)")
KWH_RE = re.compile(r"(?P<value>[-+0-9.eE]+) kWh")


@dataclass
class EpochMetrics:
    epoch: int
    metrics: dict[str, float]


@dataclass
class TaskMetrics:
    epochs: list[EpochMetrics] = field(default_factory=list)
    selected_epoch_from_log: int | None = None
    extra_final: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalRun:
    model: str
    run_id: str
    log_path: Path
    tasks: dict[str, TaskMetrics] = field(default_factory=dict)
    overall_logged: float | None = None
    kwh: float | None = None


def parse_metric_body(body: str) -> dict[str, float]:
    return {match.group("name"): float(match.group("value").rstrip(".")) for match in METRIC_RE.finditer(body)}


def discover_logs(outputs_dir: Path) -> list[Path]:
    logs = []
    for path in outputs_dir.rglob("*_eval.log"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        logs.append(path)
    return sorted(logs)


def parse_log(path: Path, outputs_dir: Path) -> EvalRun:
    rel = path.relative_to(outputs_dir)
    model = rel.parts[0]
    try:
        downstream_idx = rel.parts.index("downstream")
        run_id = "/".join(rel.parts[downstream_idx + 1 : -1])
    except ValueError:
        run_id = path.parent.name

    run = EvalRun(model=model, run_id=run_id, log_path=path)
    current_task: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            finetune = FINETUNE_RE.search(line)
            if finetune:
                current_task = finetune.group("task").lower()
                run.tasks.setdefault(current_task, TaskMetrics())
                continue

            selected = SELECTED_RE.search(line)
            if selected:
                task = selected.group("task").lower()
                run.tasks.setdefault(task, TaskMetrics()).selected_epoch_from_log = int(selected.group("epoch"))
                continue

            validation = VALID_RE.search(line)
            if validation and current_task is not None:
                metrics = parse_metric_body(validation.group("body"))
                run.tasks.setdefault(current_task, TaskMetrics()).epochs.append(
                    EpochMetrics(epoch=int(validation.group("epoch")), metrics=metrics)
                )
                continue

            extra = EXTRA_RE.search(line)
            if extra and current_task is not None:
                extra_metrics = {f"{name}_extra": value for name, value in parse_metric_body(extra.group("body")).items()}
                run.tasks.setdefault(current_task, TaskMetrics()).extra_final.update(extra_metrics)
                continue

            overall = OVERALL_RE.search(line)
            if overall:
                run.overall_logged = float(overall.group("value").rstrip("."))
                continue

            if " kWh of electricity used" in line:
                kwh = KWH_RE.search(line)
                if kwh:
                    run.kwh = float(kwh.group("value"))

    return run


def select_best(task_name: str, task_metrics: TaskMetrics) -> tuple[int | None, dict[str, float], float | None]:
    target_names = TARGET_METRICS.get(task_name, [])
    best_epoch = None
    best_metrics: dict[str, float] = {}
    best_score = None

    for epoch_metrics in task_metrics.epochs:
        available_targets = [name for name in target_names if name in epoch_metrics.metrics]
        if not available_targets:
            available_targets = list(epoch_metrics.metrics)
        if not available_targets:
            continue
        score = sum(epoch_metrics.metrics[name] for name in available_targets) / len(available_targets)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch_metrics.epoch
            best_metrics = dict(epoch_metrics.metrics)

    # Old logs only evaluated MNLI-mm once after the selected MNLI epoch. Keep it
    # as a target metric for aggregate comparability, but mark the source.
    if task_name == "mnli" and "accuracy_extra" not in best_metrics and "accuracy_extra" in task_metrics.extra_final:
        best_metrics["accuracy_extra"] = task_metrics.extra_final["accuracy_extra"]

    return best_epoch, best_metrics, best_score


def harmonic_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def summarize_run(run: EvalRun) -> dict[str, object]:
    row: dict[str, object] = {
        "model": run.model,
        "run_id": run.run_id,
        "status": "",
        "tasks_present": "",
        "tasks_missing": "",
        "best_epochs": "",
        "glue8_amean_log_best": "",
        "glue8_hmean_log_best": "",
        "overall_logged": "" if run.overall_logged is None else f"{run.overall_logged:.6f}",
        "downstream_kwh": "" if run.kwh is None else f"{run.kwh:.6f}",
        "source": str(run.log_path),
    }

    task_scores: list[float] = []
    present = []
    missing = []
    best_epochs = []

    for task in GLUE_TASKS:
        task_metrics = run.tasks.get(task)
        if task_metrics is None or len(task_metrics.epochs) == 0:
            missing.append(task)
            continue
        present.append(task)
        best_epoch, metrics, _ = select_best(task, task_metrics)
        if best_epoch is not None:
            best_epochs.append(f"{task}:{best_epoch}")

        for metric_name, metric_value in sorted(metrics.items()):
            row[f"{task}_{metric_name}_log_best"] = f"{metric_value:.6f}"

        if all(metric_name in metrics for metric_name in TARGET_METRICS[task]):
            task_scores.append(sum(metrics[name] for name in TARGET_METRICS[task]) / len(TARGET_METRICS[task]))

    row["tasks_present"] = ";".join(present)
    row["tasks_missing"] = ";".join(missing)
    row["best_epochs"] = ";".join(best_epochs)

    if len(task_scores) == len(GLUE_TASKS):
        row["status"] = "complete_log_best"
        row["glue8_amean_log_best"] = f"{sum(task_scores) / len(task_scores):.6f}"
        row["glue8_hmean_log_best"] = f"{harmonic_mean(task_scores):.6f}"
    elif present:
        row["status"] = "partial_log_best"
    else:
        row["status"] = "no_eval_metrics_in_log"

    return row


def fieldnames(rows: list[dict[str, object]]) -> list[str]:
    leading = [
        "model",
        "run_id",
        "status",
        "glue8_amean_log_best",
        "glue8_hmean_log_best",
        "tasks_present",
        "tasks_missing",
        "best_epochs",
        "overall_logged",
        "downstream_kwh",
    ]
    dynamic = sorted({key for row in rows for key in row if key not in leading and key != "source"})
    return leading + dynamic + ["source"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_as_float(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return -math.inf
    return float(value)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    complete = [row for row in rows if row["status"] == "complete_log_best"]
    complete.sort(key=lambda row: metric_as_float(row, "glue8_amean_log_best"), reverse=True)
    partial = [row for row in rows if row["status"] != "complete_log_best"]

    lines = [
        "# Best Eval From Downstream Logs",
        "",
        "Metrics are extracted from `Validation metric is ... after epoch ...` lines and selected per task using the task target metrics.",
        "For old MNLI logs that only contain final mismatched validation, `accuracy_extra` is copied from that final extra line.",
        "GLUE8 averages eight task scores; MRPC/QQP use mean accuracy and F1, STS-B uses mean Pearson and Spearman, and MNLI uses mean matched and mismatched accuracy. WNLI is omitted.",
        "",
        "## Complete GLUE8 Runs",
        "",
        "| Rank | Model | Run | GLUE8 | Hmean | Best epochs |",
        "|---:|---|---|---:|---:|---|",
    ]
    for rank, row in enumerate(complete, start=1):
        lines.append(
            f"| {rank} | `{row['model']}` | {row['run_id']} | {row['glue8_amean_log_best']} | "
            f"{row['glue8_hmean_log_best']} | {row['best_epochs']} |"
        )

    lines += [
        "",
        "## Partial Or Failed Logs",
        "",
        "| Model | Run | Status | Present | Missing |",
        "|---|---|---|---|---|",
    ]
    for row in partial:
        lines.append(f"| `{row['model']}` | {row['run_id']} | {row['status']} | {row['tasks_present']} | {row['tasks_missing']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--tables", type=Path, default=Path("tables"))
    parser.add_argument("--csv-name", default="best_eval_from_logs.csv")
    parser.add_argument("--report-name", default="best_eval_from_logs_report.md")
    args = parser.parse_args()

    logs = discover_logs(args.outputs)
    runs = [parse_log(log, args.outputs) for log in logs]
    rows = [summarize_run(run) for run in runs]

    write_csv(args.tables / args.csv_name, rows)
    write_report(args.tables / args.report_name, rows)

    complete_count = sum(1 for row in rows if row["status"] == "complete_log_best")
    print(f"Parsed {len(rows)} downstream logs; {complete_count} complete GLUE runs.")
    print(f"Wrote {args.tables / args.csv_name}")
    print(f"Wrote {args.tables / args.report_name}")


if __name__ == "__main__":
    main()
