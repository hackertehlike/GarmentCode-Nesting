"""Compute per-pattern statistics for the benchmark pattern set and save to CSV.

Produces a CSV with one row per pattern containing geometric and categorical
features that can later be joined against experiment results on `pattern_name`
for correlation / disaggregation analysis.

Usage
-----
    conda activate garment
    python -m nesting.compute_pattern_stats \\
        --patterns-file nesting-assets/patterns_100.txt \\
        --output       nesting-assets/pattern_stats_100.csv

The --patterns-file should contain one absolute path to a
*_specification.json file per line (as produced by the pattern sampler).
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from shapely.geometry import Polygon as ShapelyPolygon

warnings.filterwarnings("ignore")

import nesting.config as config
from nesting.path_extractor import PatternPathExtractor

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def poly_area(coords: List[Tuple[float, float]]) -> float:
    p = ShapelyPolygon(coords)
    return p.area if p.is_valid else p.buffer(0).area


def convexity(coords: List[Tuple[float, float]]) -> float:
    """area / convex-hull area — 1.0 = perfectly convex."""
    p = ShapelyPolygon(coords)
    if not p.is_valid:
        p = p.buffer(0)
    ch = p.convex_hull
    return p.area / ch.area if ch.area > 1e-9 else 1.0


def bbox_dims(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs, ys = zip(*coords)
    return max(xs) - min(xs), max(ys) - min(ys)


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def parse_meta(meta: dict):
    upper  = meta.get("upper", {}).get("v") or "None"
    bottom = meta.get("bottom", {}).get("v") or "None"
    wb     = bool(meta.get("wb", {}).get("v"))
    label  = f"{upper}+{bottom}" + ("+WB" if wb else "")
    return label, upper, bottom, wb


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

FIELDS = [
    "pattern_name", "category", "upper", "bottom", "has_waistband",
    "n_pieces",
    "total_area_cm2", "mean_piece_area_cm2", "std_piece_area_cm2", "cv_piece_area",
    "min_piece_area_cm2", "max_piece_area_cm2",
    "mean_convexity", "min_convexity", "std_convexity",
    "mean_aspect_ratio", "std_aspect_ratio",
    "mean_width_cm", "mean_height_cm",
]


# ---------------------------------------------------------------------------
# Per-pattern computation
# ---------------------------------------------------------------------------

def compute_stats(spec_path: Path) -> dict:
    name   = spec_path.stem.replace("_specification", "")
    folder = spec_path.parent

    extractor   = PatternPathExtractor(spec_path)
    panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
    if not panel_pieces:
        raise ValueError("no pieces extracted")

    pieces = list(panel_pieces.values())
    for p in pieces:
        p.add_seam_allowance(config.SEAM_ALLOWANCE_CM)

    dp_path = folder / f"{name}_design_params.yaml"
    meta    = {}
    if dp_path.exists():
        d    = yaml.safe_load(dp_path.read_text())
        meta = d.get("design", {}).get("meta", {})

    category, upper, bottom, wb = parse_meta(meta)

    piece_areas, piece_conv, piece_w, piece_h = [], [], [], []
    for piece in pieces:
        coords = piece.get_outer_path()
        if len(coords) < 3:
            continue
        piece_areas.append(poly_area(coords))
        piece_conv.append(convexity(coords))
        w, h = bbox_dims(coords)
        piece_w.append(w)
        piece_h.append(h)

    if not piece_areas:
        raise ValueError("no valid piece geometries")

    aspects = [w / h if h > 0 else 1.0 for w, h in zip(piece_w, piece_h)]

    return {
        "pattern_name":        name,
        "category":            category,
        "upper":               upper,
        "bottom":              bottom,
        "has_waistband":       int(wb),
        "n_pieces":            len(piece_areas),
        "total_area_cm2":      round(sum(piece_areas), 2),
        "mean_piece_area_cm2": round(np.mean(piece_areas), 2),
        "std_piece_area_cm2":  round(np.std(piece_areas), 2),
        "cv_piece_area":       round(np.std(piece_areas) / np.mean(piece_areas), 4),
        "min_piece_area_cm2":  round(min(piece_areas), 2),
        "max_piece_area_cm2":  round(max(piece_areas), 2),
        "mean_convexity":      round(np.mean(piece_conv), 4),
        "min_convexity":       round(min(piece_conv), 4),
        "std_convexity":       round(np.std(piece_conv), 4),
        "mean_aspect_ratio":   round(np.mean(aspects), 4),
        "std_aspect_ratio":    round(np.std(aspects), 4),
        "mean_width_cm":       round(np.mean(piece_w), 2),
        "mean_height_cm":      round(np.mean(piece_h), 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns-file", required=True,
                        help="Text file with one spec path per line")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    spec_paths = [Path(l.strip()) for l in
                  open(args.patterns_file).readlines() if l.strip()]
    print(f"Computing stats for {len(spec_paths)} patterns …")

    records, failed = [], []
    for spec_path in spec_paths:
        name = spec_path.stem.replace("_specification", "")
        try:
            rec = compute_stats(spec_path)
            records.append(rec)
            print(f"  [OK] {name}: {rec['n_pieces']} pcs  "
                  f"area={rec['total_area_cm2']:.0f}cm²  "
                  f"conv={rec['mean_convexity']:.3f}  "
                  f"cat={rec['category']}")
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
            failed.append(name)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)

    print(f"\nSaved {len(records)} rows → {out}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
