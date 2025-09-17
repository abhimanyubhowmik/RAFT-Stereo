#!/usr/bin/env python3
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


def parse_fog_index_from_name(path):
    # Extract trailing _fog_<num> before extension
    base = os.path.basename(path)
    m = re.search(r'_fog_(\d+)(?:\.|$)', base)
    if m:
        return int(m.group(1))
    # fallback: try fog<num>
    m = re.search(r'fog(\d+)', base)
    return int(m.group(1)) if m else None


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
        raise FileNotFoundError(f'No files match pattern: {args.pattern}')

    data = []  # (fog_idx, avg_mean_cm, sigma_cm, conf_mean, conf_sigma)
    for fp in files:
        fog_idx = parse_fog_index_from_name(fp)
        if fog_idx is None:
            continue
        avg_mean, sigma_cm, conf_mean, conf_sigma = load_csv_stats(fp)
        if np.isfinite(avg_mean) and np.isfinite(sigma_cm):
            data.append((fog_idx, avg_mean, sigma_cm, conf_mean, conf_sigma))

    if not data:
        raise RuntimeError('No valid CSV stats parsed.')

    # Sort by fog index
    data.sort(key=lambda x: x[0])

    x = [d[0] for d in data]
    depth_y = [d[1] for d in data]
    depth_err = [2.0 * d[2] for d in data]  # 2σ
    conf_y = [d[3] for d in data]
    conf_err = [2.0 * d[4] if np.isfinite(d[4]) else np.nan for d in data]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Depth bars (left axis)
    bars = ax1.bar(x, depth_y, yerr=depth_err, align='center', alpha=0.8, ecolor='black', capsize=6, color='tab:blue', label='depth mean |ΔZ| (cm)')
    ax1.set_xlabel('fog density (index)')
    ax1.set_ylabel('avg abs depth error (cm)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(axis='y', alpha=0.3)

    # Confidence markers (right axis)
    if any(np.isfinite(conf_y)):
        ax2 = ax1.twinx()
        ax2.errorbar(x, conf_y, yerr=conf_err, fmt='o-', color='tab:green', capsize=4, label='confidence mean')
        ax2.set_ylabel('confidence', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title(args.title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out)
    plt.close(fig)

    # Print a small summary
    for d in data:
        fog_idx, avg_mean, sigma_cm, conf_mean, conf_sigma = d
        conf_text = f', conf_mean={conf_mean:.3f}, conf_2sigma={2*conf_sigma:.3f}' if np.isfinite(conf_mean) and np.isfinite(conf_sigma) else ''
        print(f'fog_{fog_idx}: mean={avg_mean:.3f} cm, 2sigma={2*sigma_cm:.3f} cm{conf_text}')


if __name__ == '__main__':
    main() 