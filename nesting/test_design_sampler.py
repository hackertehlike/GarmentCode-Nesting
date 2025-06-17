#!/usr/bin/env python
# This is a test script to understand how DesignSampler works

import sys, os
from pathlib import Path

# Add the project root to sys.path for local package imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
from pygarment.garmentcode.params import DesignSampler

print("Testing DesignSampler initialization...")

# Try to create a DesignSampler with a dictionary
test_params = {"test_param": {"v": 1, "range": [0, 10], "type": "int"}}
try:
    print("Trying to create DesignSampler with a dictionary...")
    sampler1 = DesignSampler(test_params)
    print(f"Success! Sampler params: {sampler1.params}")
except Exception as e:
    print(f"Failed to create DesignSampler with a dictionary: {e}")

# Create a temporary YAML file and try to use that
import tempfile
try:
    print("\nTrying to create DesignSampler with a YAML file...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml_content = {'design': test_params}
        yaml.dump(yaml_content, tmp, default_flow_style=False)
        tmp_path = tmp.name
        print(f"Created temporary YAML file: {tmp_path}")
    
    sampler2 = DesignSampler(tmp_path)
    print(f"Success! Sampler params: {sampler2.params}")
    
    os.unlink(tmp_path)
except Exception as e:
    print(f"Failed to create DesignSampler with a YAML file: {e}")

print("\nTest completed.")
