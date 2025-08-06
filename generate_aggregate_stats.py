#!/usr/bin/env python3
"""
Generate aggregate statistics reports from all previous evolution runs.
This script analyzes the master statistics files to produce visualizations
and aggregate metrics across multiple patterns and runs.
"""

from nesting.metastatistics import MetaStatistics

if __name__ == "__main__":
    print("Generating aggregate statistics reports...")
    MetaStatistics.generate_aggregate_reports()
    print("Done!")
