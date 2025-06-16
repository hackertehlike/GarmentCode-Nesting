#!/usr/bin/env python3
import os
import json
from pathlib import Path
import sys

def extract_panel_names(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'pattern' in data and 'panels' in data['pattern']:
            panel_names = list(data['pattern']['panels'].keys())
            return panel_names
        return []
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return []

def extract_panel_order(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'pattern' in data and 'panel_order' in data['pattern']:
            return data['pattern']['panel_order']
        return []
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return []

def main():
    # Process patterns from the main patterns directory
    patterns_dir = Path("/Users/aysegulbarlas/codestuff/GarmentCode/assets/Patterns")
    
    # Also process patterns from the nesting assets directory (limited sample)
    nesting_dir = Path("/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/garmentcodedata_batch0")
    
    all_panel_names = set()
    panel_order_names = set()
    
    # Process main patterns
    for json_file in patterns_dir.glob("*_specification.json"):
        panel_names = extract_panel_names(json_file)
        panel_order = extract_panel_order(json_file)
        
        print(f"Pattern: {json_file.name}")
        print(f"  Panel names: {panel_names}")
        print(f"  Panel order: {panel_order}")
        print()
        
        all_panel_names.update(panel_names)
        panel_order_names.update(panel_order)
    
    # Process a sample of nesting patterns (first 10)
    for json_file in list(nesting_dir.glob("*_specification.json"))[:10]:
        panel_names = extract_panel_names(json_file)
        panel_order = extract_panel_order(json_file)
        
        print(f"Pattern: {json_file.name}")
        print(f"  Panel names: {panel_names}")
        print(f"  Panel order: {panel_order}")
        print()
        
        all_panel_names.update(panel_names)
        panel_order_names.update(panel_order)
    
    # Print the summary of all unique panel names
    print("\n======= SUMMARY =======")
    print(f"Total unique panel names found: {len(all_panel_names)}")
    print("All panel names:")
    for name in sorted(all_panel_names):
        print(f"  - {name}")
    
    # Check if panel_order contains any names not in panels
    extra_names = panel_order_names - all_panel_names
    if extra_names:
        print("\nNames in panel_order but not in panels:")
        for name in sorted(extra_names):
            print(f"  - {name}")
    
    # Write results to a file
    with open("/Users/aysegulbarlas/codestuff/GarmentCode/panel_names.txt", "w") as f:
        f.write("All panel names:\n")
        for name in sorted(all_panel_names):
            f.write(f"{name}\n")

if __name__ == "__main__":
    main()
