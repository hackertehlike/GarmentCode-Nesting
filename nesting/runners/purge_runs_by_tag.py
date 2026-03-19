#!/usr/bin/env python3
"""Purge per-run CSV files under nesting/experiments/runs by run_tag.

This looks for CSV files under the configured RUNS_DIR and for each file reads
the first data row (using csv.DictReader). If that row contains a 'run_tag'
field equal to the requested tag, the file is scheduled for removal.

Usage:
  python scripts/purge_runs_by_tag.py --run-tag <TAG> [--yes] [--dry-run]

By default the script runs in dry-run mode. Pass --yes to actually delete files.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
from pathlib import Path as _Path

# Ensure the repository root (parent of scripts/) is on sys.path so
# `import nesting` works when running this script directly.
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import nesting.config as config

def find_csvs(root: Path):
    if not root.exists():
        return []
    return list(root.rglob("*.csv"))


def file_has_run_tag(path: Path, tag: str) -> bool:
    """Return True if the CSV at path has run_tag == tag on its first data row.

    We read only a small portion via csv.DictReader and next(). If file is empty,
    missing the column, or unreadable we return False.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            # If run_tag not in fieldnames, quick skip
            if not reader.fieldnames or 'run_tag' not in reader.fieldnames:
                return False
            first = next(reader, None)
            if not first:
                return False
            return str(first.get('run_tag', '')).strip() == str(tag)
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Purge run CSVs by run_tag (safe dry-run by default)")
    parser.add_argument("--run-tag", required=True, dest="run_tag", help="Run tag to purge")
    parser.add_argument("--yes", action="store_true", help="Actually delete matched files (default: dry-run)")
    parser.add_argument("--runs-dir", default=config.RUNS_DIR, help="Root runs directory (default from nesting.config.RUNS_DIR)")
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_dir)
    print(f"Searching for CSVs under: {runs_root}")
    files = find_csvs(runs_root)
    print(f"Found {len(files)} CSV files to inspect")

    matched = []
    for f in files:
        if file_has_run_tag(f, args.run_tag):
            matched.append(f)

    if not matched:
        print(f"No CSV files matched run_tag '{args.run_tag}'. Nothing to do.")
        return 0

    print(f"Matched {len(matched)} files for run_tag '{args.run_tag}':")
    for p in matched:
        print(f"  {p}")

    if not args.yes:
        print("Dry-run mode: no files will be deleted. Re-run with --yes to delete.")
        return 0

    deleted = 0
    for p in matched:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {p}: {e}")

    print(f"Deleted {deleted} files")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
