#!/usr/bin/env python3
"""
Generate aggregate statistics reports from all previous evolution runs.
This script analyzes the master statistics files to produce visualizations
and aggregate metrics across multiple patterns and runs.
"""

import argparse

from nesting.metastatistics import MetaStatistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate aggregate statistics reports")
    parser.add_argument("--run-tag", action="append", dest="run_tags", help="Filter reports to specific run tag(s)")
    parser.add_argument("--config-hash", action="append", dest="config_hashes", help="Filter reports to specific config hash(es)")
    args = parser.parse_args()

    print("Generating aggregate statistics reports...")
    MetaStatistics.generate_aggregate_reports(run_tags=args.run_tags, config_hashes=args.config_hashes)
    print("Done!")
