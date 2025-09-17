#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import glob
import argparse
import csv
import numpy as np

# Headless-safe plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_fog_density_from_name(path):
    """Extract fog density value from filename, handling both integer and decimal values"""
    base = os.path.basename(path)
    
    # Try to match _fog_<decimal> pattern (e.g., _fog_0.1, _fog_1.0)
    m = re.search(r'_fog_(\d+\.?\d*)', base)
    if m:
        return float(m.group(1))
    
    # Fallback: try fog<decimal> pattern
    m = re.search(r'fog(\d+\.?\d*)', base)
    if m:
        return float(m.group(1))
    
    return None


def load_csv_stats(csv_path):
    means_cm = []
    vars_cm2 = []
    conf_means = []
    conf_vars = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m = float(row['mean_abs_err_cm'])
                v = float(row['var_abs_err_cm2'])
                cm = float(row.get('conf_mean', 'nan'))
                cv = float(row.get('conf_var', 'nan'))
                if np.isfinite(m) and np.isfinite(v):
                    means_cm.append(m)
                    vars_cm2.append(v)
                if np.isfinite(cm) and np.isfinite(cv):
                    conf_means.append(cm)
                    conf_vars.append(cv)
            except Exception:
                continue
    avg_mean = float(np.nanmean(means_cm)) if len(means_cm) else np.nan
    sigma_cm = float(np.sqrt(np.nanmean(np.maximum(vars_cm2, 0.0)))) if len(vars_cm2) else np.nan
    conf_mean = float(np.nanmean(conf_means)) if len(conf_means) else np.nan
    conf_sigma = float(np.sqrt(np.nanmean(np.maximum(conf_vars, 0.0)))) if len(conf_vars) else np.nan
    return avg_mean, sigma_cm, conf_mean, conf_sigma


def main():
    parser = argparse.ArgumentParser(description='Line plot: fog density vs avg depth error (cm) and confidence with 2σ error bars.')
    parser.add_argument('--pattern', default='eval_output/depth_error_log_fog_*.csv', help='Glob pattern for CSV files')
    parser.add_argument('--out', default='eval_output/depth_error_fog_line.png', help='Output PNG path')
    parser.add_argument('--title', default='Average Absolute Depth Error and Confidence vs Fog Density', help='Plot title')
    parser.add_argument('--figsize', nargs=2, type=float, default=[14, 8], help='Figure size as width height (default: 14 8)')
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError('No files match pattern: {}'.format(args.pattern))

    data = []  # (fog_density, avg_mean_cm, sigma_cm, conf_mean, conf_sigma)
    for fp in files:
        fog_density = parse_fog_density_from_name(fp)
        if fog_density is None:
            print("Warning: Could not parse fog density from {}, skipping".format(fp))
            continue
        avg_mean, sigma_cm, conf_mean, conf_sigma = load_csv_stats(fp)
        if np.isfinite(avg_mean) and np.isfinite(sigma_cm):
            data.append((fog_density, avg_mean, sigma_cm, conf_mean, conf_sigma))

    if not data:
        raise RuntimeError('No valid CSV stats parsed.')

    # Sort by fog density
    data.sort(key=lambda x: x[0])

    x = [d[0] for d in data]
    depth_y = [d[1] for d in data]
    depth_err = [2.0 * d[2] for d in data]  # 2σ
    conf_y = [d[3] for d in data]
    conf_err = [2.0 * d[4] if np.isfinite(d[4]) else np.nan for d in data]

    # Create figure with larger size for better readability
    fig, ax1 = plt.subplots(figsize=args.figsize)

    # Depth error line plot (left axis) - Blue with solid line
    line1 = ax1.plot(x, depth_y, 'o-', color='tab:blue', linewidth=2.5, markersize=6, 
                     label='Depth Error Mean |ΔZ| (cm)', markerfacecolor='tab:blue', 
                     markeredgecolor='white', markeredgewidth=1)
    
    # Depth error variance - Blue with dotted line and light fill
    ax1.fill_between(x, 
                     [y - err for y, err in zip(depth_y, depth_err)], 
                     [y + err for y, err in zip(depth_y, depth_err)], 
                     color='tab:blue', alpha=0.15, linestyle='--', linewidth=1.5,
                     label='Depth Error ±2σ (dotted)')
    
    ax1.set_xlabel('Fog Density', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Absolute Depth Error (cm)', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set x-axis ticks to show fog density values
    ax1.set_xticks(x)
    ax1.set_xticklabels(['{:.1f}'.format(val) for val in x])

    # Confidence line plot (right axis) - Green with different line style
    if any(np.isfinite(conf_y)):
        ax2 = ax1.twinx()
        
        # Confidence mean line - Green with dash-dot line
        line2 = ax2.plot(x, conf_y, 's-', color='tab:green', linewidth=2.5, markersize=6,
                         label='Confidence Mean', markerfacecolor='tab:green',
                         markeredgecolor='white', markeredgewidth=1, linestyle='-.')
        
        # Confidence variance - Green with dashed line and different fill pattern
        ax2.fill_between(x, 
                         [y - err if np.isfinite(err) else y for y, err in zip(conf_y, conf_err)], 
                         [y + err if np.isfinite(err) else y for y, err in zip(conf_y, conf_err)], 
                         color='tab:green', alpha=0.2, linestyle=':', linewidth=2,
                         label='Confidence ±2σ (dashed)')
        
        ax2.set_ylabel('Confidence', color='tab:green', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=10)

    # Add legend with both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if any(np.isfinite(conf_y)):
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                  bbox_to_anchor=(0.02, 0.98), fontsize=10, framealpha=0.9)
    else:
        ax1.legend(lines1, labels1, loc='upper left', fontsize=10, framealpha=0.9)

    plt.title(args.title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Print a small summary
    print("\nProcessed {} fog density experiments:".format(len(data)))
    for d in data:
        fog_density, avg_mean, sigma_cm, conf_mean, conf_sigma = d
        conf_text = ', conf_mean={:.3f}, conf_2sigma={:.3f}'.format(conf_mean, 2*conf_sigma) if np.isfinite(conf_mean) and np.isfinite(conf_sigma) else ''
        print('fog_{:.1f}: mean={:.3f} cm, 2sigma={:.3f} cm{}'.format(fog_density, avg_mean, 2*sigma_cm, conf_text))


if __name__ == '__main__':
    main()
