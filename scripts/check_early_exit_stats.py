#!/usr/bin/env python
"""Check early-exit convergence tables for non-trivial adaptive depth."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys


def latest_table(base_dir: str, run_name: str) -> str:
    pattern = os.path.join(base_dir, run_name, "**", f"table_{run_name}_convergence_results.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No convergence table matched {pattern}")
    return max(matches, key=os.path.getmtime)


def read_last_value(path: str, column: str) -> float:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row.get(column) not in (None, "")]
    if not rows:
        raise ValueError(f"Column {column!r} is missing or empty in {path}")
    return float(rows[-1][column])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help="Cramming run name to inspect under base-dir.")
    parser.add_argument("--base-dir", default="outputs", help="Cramming base output directory.")
    parser.add_argument("--column", default="exit/max_depth_frac", help="Convergence-table column to check.")
    parser.add_argument("--max-allowed", type=float, default=0.95, help="Fail when column is greater than or equal to this value.")
    args = parser.parse_args()

    table = latest_table(args.base_dir, args.run_name)
    value = read_last_value(table, args.column)
    print(f"{args.run_name}: {args.column}={value:.4f} from {table}")
    if value >= args.max_allowed:
        print(f"FAILED: {args.column} >= {args.max_allowed:.4f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
