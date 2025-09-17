#!/usr/bin/env python3
"""
Demo script to generate plots with different sigma multipliers
"""

import subprocess
import sys

def run_plot(sigma_multiplier, output_name):
    """Generate a plot with specific sigma multiplier"""
    cmd = [
        'python3', 
        'plot_fog_bar.py',
        '--pattern=eval_output/depth_error_log_fog_*.csv',
        '--out=eval_output/{}'.format(output_name),
        '--title=Depth Error Analysis with ±{:.1f}σ Error Bars'.format(sigma_multiplier),
        '--sigma-multiplier', str(sigma_multiplier)
    ]
    
    print("Generating plot with ±{:.1f}σ error bars...".format(sigma_multiplier))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Successfully created: {}".format(output_name))
    else:
        print("✗ Failed to create: {}".format(output_name))
        print("Error: {}".format(result.stderr))
    
    return result.returncode == 0

def main():
    print("Generating demonstration plots with different sigma multipliers...")
    
    # Different sigma multipliers to demonstrate
    sigma_configs = [
        (0.5, "demo_0.5sigma.png"),
        (1.0, "demo_1.0sigma.png"), 
        (2.0, "demo_2.0sigma.png"),
        (3.0, "demo_3.0sigma.png")
    ]
    
    success_count = 0
    for sigma, filename in sigma_configs:
        if run_plot(sigma, filename):
            success_count += 1
    
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print("Successfully generated {} out of {} plots".format(success_count, len(sigma_configs)))
    print("Check the eval_output/ directory for the generated plots:")
    for _, filename in sigma_configs:
        print("  - {}".format(filename))
    
    return 0 if success_count == len(sigma_configs) else 1

if __name__ == '__main__':
    sys.exit(main())
