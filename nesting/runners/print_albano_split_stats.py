#!/usr/bin/env python3
"""
Print split vs no-split stats for 'albano' rows from final_metrics_legacy.csv

Metrics printed per group (splits, nosplits):
- max bb usage
- average bb usage
- max rest length (400 - rest_length_cm)
- average rest length (400 - rest_length_cm)
- max combined (bb_with_rest_length)
- average combined (bb_with_rest_length)
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "nesting" / "experiments" / "aggregate" / "final_metrics_legacy.csv"


def to_float(s):
    try:
        return float(s)
    except Exception:
        return None


def stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    max_v = max(vals)
    avg_v = sum(vals) / len(vals)
    return max_v, avg_v


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    split_usage = []
    split_rest_inv = []  # 400 - rest_length
    split_combined = []

    nosplit_usage = []
    nosplit_rest_inv = []
    nosplit_combined = []

    with SRC.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("pattern_name") or "").strip() != "albano":
                continue
            cfg = (row.get("config_hash") or "").strip()
            rest_len = to_float(row.get("rest_length_cm"))
            usage_bb = to_float(row.get("usage_bb"))
            combined = to_float(row.get("bb_with_rest_length"))
            rest_inv = (400.0 - rest_len) if rest_len is not None else None

            if cfg == "splits":
                split_usage.append(usage_bb)
                split_rest_inv.append(rest_inv)
                split_combined.append(combined)
            elif cfg == "nosplits":
                nosplit_usage.append(usage_bb)
                nosplit_rest_inv.append(rest_inv)
                nosplit_combined.append(combined)

    s_max_u, s_avg_u = stats(split_usage)
    s_max_r, s_avg_r = stats(split_rest_inv)
    s_max_c, s_avg_c = stats(split_combined)

    n_max_u, n_avg_u = stats(nosplit_usage)
    n_max_r, n_avg_r = stats(nosplit_rest_inv)
    n_max_c, n_avg_c = stats(nosplit_combined)

    print(
        "Split: max bb usage: ", s_max_u,
        " average bb usage: ", s_avg_u,
        " max rest length: ", s_max_r,
        " average rest length: ", s_avg_r,
        " max combined: ", s_max_c,
        " average combined: ", s_avg_c,
    )

    print(
        "No Split: max bb usage: ", n_max_u,
        " average bb usage: ", n_avg_u,
        " max rest length: ", n_max_r,
        " average rest length: ", n_avg_r,
        " max combined: ", n_max_c,
        " average combined: ", n_avg_c,
    )


if __name__ == "__main__":
    main()
