#!/usr/bin/env python3
"""
Extract per-pattern averages by config from the existing aggregate CSV.

Reads: nesting/experiments/aggregate/per-pattern-final-by-config.csv
Writes: nesting/experiments/aggregate/per-pattern-average-by-config.csv

Keeps columns: pattern_name, config_hash, and all columns ending with '_mean'.
Formats numeric means to 2 decimals by default, except specific utilization
columns which are formatted to 4 decimals.
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

src = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "per-pattern-final-by-config.csv"
dst = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "per-pattern-average-by-config.csv"

def main():
    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Select keys: pattern_name, config_hash, and any *_mean columns, excluding bb_cc_area_mean
        mean_cols = [c for c in fieldnames if c.endswith("_mean") and c != "bb_cc_area_mean"]
        out_fields = ["pattern_name", "config_hash"] + mean_cols

        # Columns to keep at 4 decimal precision
        four_decimal_cols = {
            "usage_bb_mean",
            "concave_hull_utilization_mean",
            "bb_with_rest_length_mean",
        }

        with dst.open("w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=out_fields)
            writer.writeheader()
            for row in reader:
                out_row = {}
                for k in out_fields:
                    val = row.get(k, "")
                    if k.endswith("_mean") and val not in ("", None):
                        try:
                            num = float(val)
                            if k in four_decimal_cols:
                                val = f"{num:.4f}"
                            else:
                                val = f"{num:.2f}"
                        except (ValueError, TypeError):
                            # Leave as-is if not a float
                            pass
                    out_row[k] = val
                writer.writerow(out_row)

    print(f"Wrote {dst}")

if __name__ == "__main__":
    main()
