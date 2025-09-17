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


def parse_fog_and_depth(path):
    base = os.path.basename(path)
    fog = None
    depth = None
    m = re.search(r'_fog_(\d+\.?\d*)', base)
    if m:
        fog = float(m.group(1))
    m = re.search(r'_depth_(\d+\.?\d*)', base)
    if m:
        depth = float(m.group(1))
    return fog, depth


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
    parser = argparse.ArgumentParser(description='Multi-series plot: Depth vs avg depth error (cm) and confidence per fog density with σ error bands.')
    parser.add_argument('--pattern', default='eval_output/depth_error_log_fog_*_depth_*.csv', help='Glob pattern for CSV files')
    parser.add_argument('--out', default='eval_output/depth_error_depth_multi.png', help='Output PNG path')
    parser.add_argument('--title', default='Average Absolute Depth Error and Confidence vs Depth (colored by fog density)', help='Plot title')
    parser.add_argument('--figsize', nargs=2, type=float, default=[14, 8], help='Figure size as width height')
    parser.add_argument('--sigma-multiplier', type=float, default=1.0, help='Sigma multiplier for error bands')
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError('No files match pattern: {}'.format(args.pattern))

    # Group by fog density
    fog_to_records = {}
    for fp in files:
        fog, depth = parse_fog_and_depth(fp)
        if fog is None or depth is None:
            continue
        avg_mean, sigma_cm, conf_mean, conf_sigma = load_csv_stats(fp)
        if np.isfinite(avg_mean) and np.isfinite(sigma_cm):
            fog_to_records.setdefault(fog, []).append((depth, avg_mean, sigma_cm, conf_mean, conf_sigma))

    if not fog_to_records:
        raise RuntimeError('No valid CSV stats parsed.')

    # Colors cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])

    fig, ax1 = plt.subplots(figsize=args.figsize)

    # Left axis: depth error
    for idx, (fog, records) in enumerate(sorted(fog_to_records.items(), key=lambda kv: kv[0])):
        records.sort(key=lambda r: r[0])
        x = [r[0] for r in records]
        depth_err_mean = [r[1] for r in records]
        depth_err_band = [args.sigma_multiplier * r[2] for r in records]
        color = color_cycle[idx % len(color_cycle)]
        ax1.plot(x, depth_err_mean, 'o-', color=color, linewidth=2.5, markersize=6,
                 label='Depth Error |ΔZ| (cm), fog {:.1f}'.format(fog),
                 markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
        ax1.fill_between(x,
                         [yy - ee for yy, ee in zip(depth_err_mean, depth_err_band)],
                         [yy + ee for yy, ee in zip(depth_err_mean, depth_err_band)],
                         color=color, alpha=0.15, linestyle='--', linewidth=1.5)

    ax1.set_xlabel('Depth', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Absolute Depth Error (cm)', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # x ticks combined from all depths seen
    all_depths = sorted({d for recs in fog_to_records.values() for d, *_ in recs})
    ax1.set_xticks(all_depths)
    ax1.set_xticklabels(['{:.1f}'.format(val) for val in all_depths])

    # Right axis: confidence
    ax2 = ax1.twinx()
    for idx, (fog, records) in enumerate(sorted(fog_to_records.items(), key=lambda kv: kv[0])):
        records.sort(key=lambda r: r[0])
        x = [r[0] for r in records]
        conf_mean = [r[3] for r in records]
        conf_band = [args.sigma_multiplier * r[4] if np.isfinite(r[4]) else np.nan for r in records]
        color = color_cycle[idx % len(color_cycle)]
        ax2.plot(x, conf_mean, 's-.', color=color, linewidth=2.0, markersize=5,
                 label='Confidence, fog {:.1f}'.format(fog),
                 markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
        ax2.fill_between(x,
                         [yy - ee if np.isfinite(ee) else yy for yy, ee in zip(conf_mean, conf_band)],
                         [yy + ee if np.isfinite(ee) else yy for yy, ee in zip(conf_mean, conf_band)],
                         color=color, alpha=0.12, linestyle=':', linewidth=1.2)

    ax2.set_ylabel('Confidence', color='tab:green', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=10)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.9)

    title_with_sigma = '{} (±{:.1f}σ bands)'.format(args.title, args.sigma_multiplier)
    plt.title(title_with_sigma, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print('Saved plot to {}'.format(args.out))


if __name__ == '__main__':
    main() 