#!/usr/bin/env python3
"""
Test script for fog density experiment automation
This runs a quick test with just 2 fog densities and shorter recording time
"""

import subprocess
import sys

def main():
    print("Running quick test of fog density experiment automation...")
    print("This will test with fog densities 0.1 and 0.5, 20 seconds each")
    
    cmd = [
        'python3', 
        'fog_experiment_automation.py',
        '--fog-densities', '0.1', '0.5',
        '--recording-time', '20',
        '--wait-time', '3',
        '--debug'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Test completed successfully!")
        else:
            print("✗ Test failed with return code: {}".format(result.returncode))
            
    except Exception as e:
        print("Error running test: {}".format(e))
        return 1
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
