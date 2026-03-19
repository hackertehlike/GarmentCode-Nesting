#!/usr/bin/env python3
"""Convenience script to purge all GA aggregate CSV/stat files.
Usage:
  python scripts/purge_aggregate_stats.py            # purge + regenerate headers
  python scripts/purge_aggregate_stats.py --reports  # purge, then regenerate reports (will be empty)
"""
from nesting.analysis.metastatistics import MetaStatistics, MetaStatisticsCLI
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', action='store_true', help='Also run report generation after purge')
    args = parser.parse_args()
    MetaStatistics.purge_all()
    if args.reports:
        MetaStatisticsCLI.generate_reports()

if __name__ == '__main__':
    main()
