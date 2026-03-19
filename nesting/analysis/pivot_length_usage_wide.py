#!/usr/bin/env python3
"""
Pivot per-pattern averages into a wide table with per-config columns.

Reads: nesting/experiments/aggregate/per-pattern-average-by-config.csv
Writes: nesting/experiments/aggregate/per-pattern-length-usage-wide.csv

Row index: pattern_name
Columns: For each config_hash, two columns:
  - "<config_hash> solution length" where value = 800 - rest_length_cm_mean (formatted .2f)
  - "<config_hash> usage bb" where value = usage_bb_mean (formatted .4f)
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SRC = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "per-pattern-average-by-config.csv"
DST = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "per-pattern-length-usage-wide.csv"


def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source CSV not found: {SRC}")

    # Collect data indexed by pattern_name and set of config hashes
    by_pattern = {}
    config_hashes = []
    seen_hash = set()

    with SRC.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = row.get("pattern_name", "").strip()
            cfg = row.get("config_hash", "").strip()
            if not pattern or not cfg:
                continue

            if cfg not in seen_hash:
                seen_hash.add(cfg)
                config_hashes.append(cfg)

            rest_len = to_float(row.get("rest_length_cm_mean"))
            usage_bb = to_float(row.get("usage_bb_mean"))

            if pattern not in by_pattern:
                by_pattern[pattern] = {}
            by_pattern[pattern][cfg] = (rest_len, usage_bb)

    # Prepare header
    header = ["pattern_name"]
    for cfg in config_hashes:
        header.append(f"{cfg} solution length")
        header.append(f"{cfg} usage bb")

    # Write output
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for pattern in sorted(by_pattern.keys()):
            row = [pattern]
            vals = by_pattern[pattern]
            for cfg in config_hashes:
                rest_len, usage_bb = vals.get(cfg, (None, None))
                # solution length = 800 - rest_length_cm_mean
                if rest_len is None:
                    sol_len_str = ""
                else:
                    sol_len = 800.0 - rest_len
                    sol_len_str = f"{sol_len:.2f}"

                if usage_bb is None:
                    usage_bb_str = ""
                else:
                    usage_bb_str = f"{usage_bb:.4f}"

                row.extend([sol_len_str, usage_bb_str])
            writer.writerow(row)

    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
