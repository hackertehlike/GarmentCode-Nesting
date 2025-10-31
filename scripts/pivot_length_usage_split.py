#!/usr/bin/env python3
"""
Generate two wide CSVs from per-pattern averages with renamed per-config columns.

Reads: nesting/experiments/aggregate/per-pattern-average-by-config.csv
Writes:
  - nesting/experiments/aggregate/per-pattern-length-wide.csv
  - nesting/experiments/aggregate/per-pattern-usage-bb-wide.csv

Rows: pattern_name
Columns:
  - For solution length: one column per config label, value = 800 - rest_length_cm_mean (formatted .2f)
  - For usage bb: one column per config label, value = usage_bb_mean (formatted .4f)

Config label mapping provided by user:
  05dfa5a5 -> GA sticky
  102ad642 -> NE - dp
  11820f60 -> NE - split
  64e50bae -> NE - dp, split
  90d92aab -> GA lex
  9b8a78c8 -> NE - rot
  d65f4688 -> NE (full)
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SRC = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "per-pattern-average-by-config.csv"
DST_LEN = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "per-pattern-length-wide.csv"
DST_USAGE = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "per-pattern-usage-bb-wide.csv"

LABELS = {
    "05dfa5a5": "GA sticky",
    "102ad642": "NE - dp",
    "11820f60": "NE - split",
    "64e50bae": "NE - dp, split",
    "90d92aab": "GA lex",
    "9b8a78c8": "NE - rot",
    "d65f4688": "NE (full)",
}


def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source CSV not found: {SRC}")

    by_pattern = {}
    config_order = []
    seen = set()

    with SRC.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = (row.get("pattern_name") or "").strip()
            cfg = (row.get("config_hash") or "").strip()
            if not pattern or not cfg:
                continue

            if cfg not in seen:
                seen.add(cfg)
                config_order.append(cfg)

            rest_len = to_float(row.get("rest_length_cm_mean"))
            usage_bb = to_float(row.get("usage_bb_mean"))

            if pattern not in by_pattern:
                by_pattern[pattern] = {}
            by_pattern[pattern][cfg] = (rest_len, usage_bb)

    # Build header columns using mapping where available; fallback to hash
    labels_in_order = [LABELS.get(cfg, cfg) for cfg in config_order]

    # Write length CSV
    DST_LEN.parent.mkdir(parents=True, exist_ok=True)
    with DST_LEN.open("w", newline="") as f_len:
        writer = csv.writer(f_len)
        writer.writerow(["pattern_name", *labels_in_order])
        for pattern in sorted(by_pattern.keys()):
            vals_by_cfg = by_pattern[pattern]
            row = [pattern]
            for cfg in config_order:
                rest_len, _ = vals_by_cfg.get(cfg, (None, None))
                if rest_len is None:
                    row.append("")
                else:
                    sol_len = 800.0 - rest_len
                    row.append(f"{sol_len:.2f}")
            writer.writerow(row)

    # Write usage bb CSV
    with DST_USAGE.open("w", newline="") as f_usage:
        writer = csv.writer(f_usage)
        writer.writerow(["pattern_name", *labels_in_order])
        for pattern in sorted(by_pattern.keys()):
            vals_by_cfg = by_pattern[pattern]
            row = [pattern]
            for cfg in config_order:
                _, usage_bb = vals_by_cfg.get(cfg, (None, None))
                if usage_bb is None:
                    row.append("")
                else:
                    row.append(f"{usage_bb:.4f}")
            writer.writerow(row)

    print(f"Wrote {DST_LEN}")
    print(f"Wrote {DST_USAGE}")


if __name__ == "__main__":
    main()
