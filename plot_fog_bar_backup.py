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
    parser = argparse.ArgumentParser(description='Bar plot: fog density vs avg depth error (cm) and confidence with 2σ error bars.')
    parser.add_argument('--pattern', default='eval_output/depth_error_log_fog_*.csv', help='Glob pattern for CSV files')
    parser.add_argument('--out', default='eval_output/depth_error_fog_bar.png', help='Output PNG path')
    parser.add_argument('--title', default='Average Absolute Depth Error and Confidence vs Fog Density', help='Plot title')
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

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Depth bars (left axis)
    bars = ax1.bar(x, depth_y, yerr=depth_err, align='center', alpha=0.8, ecolor='black', capsize=6, color='tab:blue', label='depth mean |ΔZ| (cm)')
    ax1.set_xlabel('fog density')
    ax1.set_ylabel('avg abs depth error (cm)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(axis='y', alpha=0.3)
    
    # Set x-axis ticks to show fog density values
    ax1.set_xticks(x)
    ax1.set_xticklabels(['{:.1f}'.format(val) for val in x])

    # Confidence markers (right axis)
    if any(np.isfinite(conf_y)):
        ax2 = ax1.twinx()
        ax2.errorbar(x, conf_y, yerr=conf_err, fmt='o-', color='tab:green', capsize=4, label='confidence mean')
        ax2.set_ylabel('confidence', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title(args.title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Print a small summary
    print("\nProcessed {} fog density experiments:".format(len(data)))
    for d in data:
        fog_density, avg_mean, sigma_cm, conf_mean, conf_sigma = d
        conf_text = ', conf_mean={:.3f}, conf_2sigma={:.3f}'.format(conf_mean, 2*conf_sigma) if np.isfinite(conf_mean) and np.isfinite(conf_sigma) else ''
        print('fog_{:.1f}: mean={:.3f} cm, 2sigma={:.3f} cm{}'.format(fog_density, avg_mean, 2*sigma_cm, conf_text))


if __name__ == '__main__':
    main()
