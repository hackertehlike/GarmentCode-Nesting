#!/usr/bin/env python3
"""Purge run directories under nesting/experiments/runs by timestamp in the name.

Directory names are expected to end with a timestamp token after the last underscore,
for example: <pattern>_20240905 or <pattern>_20240905_1504. The script extracts the
first 8 digits of that trailing token as YYYYMMDD and compares it to the provided
date(s).

By default the script performs a dry-run and only lists matching directories. Use
--yes to actually delete them.
"""
from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path as _Path

# Make repository root importable when run from scripts/
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import nesting.config as config


def find_candidate_dirs(root: _Path) -> list[_Path]:
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def extract_yyyymmdd_from_name(name: str) -> str | None:
        """Return the first 8-digit YYYYMMDD string found anywhere in the name.

        This handles names like:
            rand_0883WEOFDQ_20250901_235820 -> '20250901'
            pattern_20240905 -> '20240905'
        """
        m = re.search(r"(\d{8})", name)
        if m:
                return m.group(1)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Purge run directories by trailing YYYYMMDD timestamp (dry-run by default)")
    parser.add_argument("--after", type=int, help="Purge directories with timestamp strictly after this YYYYMMDD value", required=True)
    parser.add_argument("--before", type=int, help="Optional: only purge directories with timestamp strictly before this YYYYMMDD value")
    parser.add_argument("--runs-dir", default=config.RUNS_DIR, help="Root runs directory (default from nesting.config.RUNS_DIR)")
    parser.add_argument("--yes", action="store_true", help="Actually delete matched directories (default: dry-run)")
    args = parser.parse_args(argv)

    runs_root = _Path(args.runs_dir)
    print(f"Scanning run directories under: {runs_root}")

    candidates = find_candidate_dirs(runs_root)
    print(f"Found {len(candidates)} directories to inspect")

    matched: list[_Path] = []
    for d in candidates:
        ts = extract_yyyymmdd_from_name(d.name)
        if not ts:
            continue
        try:
            ts_val = int(ts)
        except ValueError:
            continue
        if ts_val <= int(args.after):
            continue
        if args.before and ts_val >= int(args.before):
            continue
        matched.append(d)

    if not matched:
        print(f"No directories matched criteria (after={args.after}, before={args.before})")
        return 0

    print(f"Matched {len(matched)} directories (after={args.after}):")
    for p in matched:
        print(f"  {p}")

    if not args.yes:
        print("Dry-run mode: no directories were deleted. Re-run with --yes to delete.")
        return 0

    deleted = 0
    for p in matched:
        try:
            shutil.rmtree(p)
            deleted += 1
        except Exception as e:
            print(f"Failed to remove {p}: {e}")

    print(f"Deleted {deleted} directories")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
