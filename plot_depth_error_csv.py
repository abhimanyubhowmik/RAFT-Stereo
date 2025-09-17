#!/usr/bin/env python3
import os
import argparse
import csv
import math
import numpy as np

# Headless-safe plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_depth_error_csv(csv_path):
    stamps = []
    means_cm = []
    vars_cm2 = []
    counts = []
    densities = []
    conf_means = []
    conf_vars = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Expected headers: stamp, mean_abs_err_cm, var_abs_err_cm2, count, density, [optional conf_mean, conf_var]
        for row in reader:
            try:
                stamps.append(float(row['stamp']))
                means_cm.append(float(row['mean_abs_err_cm']))
                vars_cm2.append(float(row['var_abs_err_cm2']))
                counts.append(int(row.get('count', '0')))
                densities.append(float(row.get('density', 'nan')))
                if 'conf_mean' in row and 'conf_var' in row:
                    conf_means.append(float(row['conf_mean']))
                    conf_vars.append(float(row['conf_var']))
            except Exception:
                continue

    stamps = np.array(stamps, dtype=np.float64)
    means_cm = np.array(means_cm, dtype=np.float64)
    vars_cm2 = np.array(vars_cm2, dtype=np.float64)
    counts = np.array(counts, dtype=np.int64)
    densities = np.array(densities, dtype=np.float64)
    conf_means = np.array(conf_means, dtype=np.float64) if len(conf_means) else None
    conf_vars = np.array(conf_vars, dtype=np.float64) if len(conf_vars) else None
    return stamps, means_cm, vars_cm2, counts, densities, conf_means, conf_vars


def plot_error_with_variance(stamps, means_cm, vars_cm2, out_path, sigma_multiplier=2.0, title_suffix='', conf_means=None, conf_vars=None):
    if stamps.size == 0:
        raise ValueError('No data to plot.')

    # Convert to relative time (seconds since first sample)
    xs = stamps - stamps[0]

    # Depth variance band
    sigma_cm = np.sqrt(np.maximum(vars_cm2, 0.0))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Depth (left axis)
    ax1.fill_between(xs, means_cm - sigma_multiplier * sigma_cm, means_cm + sigma_multiplier * sigma_cm,
                     color='tab:blue', alpha=0.2, label=f'depth ±{int(sigma_multiplier)}σ')
    l1, = ax1.plot(xs, means_cm, color='tab:blue', marker='o', markersize=2, linewidth=1.0, label='depth mean |ΔZ|')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('depth error (cm)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    # Confidence (right axis)
    lines = [l1]
    labels = ['depth mean |ΔZ|']
    if conf_means is not None and conf_vars is not None and conf_means.size == stamps.size:
        ax2 = ax1.twinx()
        conf_sigma = np.sqrt(np.maximum(conf_vars, 0.0))
        ax2.fill_between(xs, conf_means - sigma_multiplier * conf_sigma, conf_means + sigma_multiplier * conf_sigma,
                         color='tab:green', alpha=0.2, label=f'conf ±{int(sigma_multiplier)}σ')
        l2, = ax2.plot(xs, conf_means, color='tab:green', marker='x', markersize=2, linewidth=1.0, label='conf mean')
        ax2.set_ylabel('confidence', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        lines.append(l2)
        labels.append('conf mean')

    plt.title(f'Depth Error and Confidence over Time{(" - " + title_suffix) if title_suffix else ""}')
    fig.legend(lines, labels, loc='upper right')
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot depth error CSV with variance band and optional confidence.')
    parser.add_argument('--csv', required=True, help='Path to depth_error_log.csv')
    parser.add_argument('--out', default='depth_error_from_csv.png', help='Output PNG path')
    parser.add_argument('--sigma', type=float, default=2.0, help='Sigma multiplier for variance bands (2 or 3)')
    parser.add_argument('--title', default='', help='Optional title suffix')
    args = parser.parse_args()

    stamps, means_cm, vars_cm2, counts, densities, conf_means, conf_vars = load_depth_error_csv(args.csv)
    plot_error_with_variance(stamps, means_cm, vars_cm2, args.out, sigma_multiplier=args.sigma, title_suffix=args.title, conf_means=conf_means, conf_vars=conf_vars)


if __name__ == '__main__':
    main() 