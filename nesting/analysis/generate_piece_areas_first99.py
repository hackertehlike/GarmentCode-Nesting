#!/usr/bin/env python3
"""
Generate a CSV with the total area of pieces (with 1 cm seam allowance)
for the first 99 patterns under nesting-assets/pattern_files.

Output: nesting/experiments/aggregate/piece_areas_first99_sa1cm.csv
Columns: pattern_name,total_area_cm2,piece_count
"""
import csv
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nesting.data.path_extractor import PatternPathExtractor
from nesting.core import utils


def find_spec_json(pattern_dir: Path) -> Optional[Path]:
    """Return the specification.json path inside a pattern directory."""
    # Prefer *_specification.json
    candidates = sorted(pattern_dir.glob("*_specification.json"))
    if candidates:
        return candidates[0]
    # Fallback: any .json in the folder
    any_json = sorted(pattern_dir.glob("*.json"))
    return any_json[0] if any_json else None


def compute_total_area_with_sa(spec_path: Path, seam_allowance_cm: float = 1.0, samples_per_edge: int = 10) -> tuple[float, int]:
    """Load pattern, offset each piece by seam_allowance_cm, sum outer-path areas.

    Returns (total_area_cm2, piece_count).
    """
    extractor = PatternPathExtractor(str(spec_path))
    pieces = extractor.get_all_panel_pieces(samples_per_edge=samples_per_edge)
    total_area = 0.0
    for piece in pieces.values():
        # Apply seam allowance to compute final sewing outline
        piece.add_seam_allowance(seam_allowance_cm)
        # Area of the resulting outer path (cm^2)
        total_area += utils.polygon_area(piece.get_outer_path())
    return total_area, len(pieces)


def main():
    root = REPO_ROOT / "nesting-assets" / "pattern_files"
    if not root.exists():
        print(f"Pattern files directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Sort dirs deterministically and take first 99
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name != ".DS_Store"]
    dirs = dirs[:99]

    out_dir = REPO_ROOT / "nesting" / "experiments" / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "piece_areas_first99_sa1cm.csv"

    rows = []
    for d in dirs:
        spec = find_spec_json(d)
        if not spec:
            # Skip if no spec json
            continue
        try:
            total_area, count = compute_total_area_with_sa(spec, seam_allowance_cm=1.0, samples_per_edge=10)
            rows.append({
                "pattern_name": d.name,
                "total_area_cm2": f"{total_area:.6f}",
                "theoretical_lower_bound": f"{(total_area/140.0):.6f}",
                "piece_count": count,
            })
        except Exception as e:
            # Keep going; record error with zero area
            rows.append({
                "pattern_name": d.name,
                "total_area_cm2": "",
                "theoretical_lower_bound": "",
                "piece_count": 0,
                # Optional: could add an error column if desired
            })

    # Write CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern_name", "total_area_cm2", "theoretical_lower_bound", "piece_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} records to {out_csv}")


if __name__ == "__main__":
    main()
