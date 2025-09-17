# Plot Fog Bar Script Usage Guide

## Overview

The `plot_fog_bar.py` script creates line plots comparing fog density vs depth error and confidence with customizable sigma error bars. The script has been updated to address the following improvements:

1. **Line Charts**: Both depth error and confidence are now displayed as line charts instead of bar charts
2. **Distinct Variance Boundaries**: Different line styles and fill patterns prevent overlapping of variance boundaries
3. **Customizable Sigma Multiplier**: Control the sigma boundary (1σ, 2σ, 0.5σ, etc.)

## Key Features

- **Line Plot Style**: Both depth error and confidence use line charts with different markers
- **Distinct Variance Visualization**: 
  - Depth error: Blue dotted lines with light fill
  - Confidence: Green dashed lines with different fill pattern
- **Customizable Sigma**: Control error bar coverage with `--sigma-multiplier` parameter
- **Dual Y-Axes**: Left axis for depth error (cm), right axis for confidence
- **Professional Styling**: High-resolution output with proper legends and formatting

## Usage

### Basic Usage
```bash
python3 plot_fog_bar.py
```

### Custom Sigma Multipliers

**1σ (68% confidence interval):**
```bash
python3 plot_fog_bar.py --sigma-multiplier 1.0
```

**2σ (95% confidence interval) - Default:**
```bash
python3 plot_fog_bar.py --sigma-multiplier 2.0
```

**0.5σ (38% confidence interval):**
```bash
python3 plot_fog_bar.py --sigma-multiplier 0.5
```

**3σ (99.7% confidence interval):**
```bash
python3 plot_fog_bar.py --sigma-multiplier 3.0
```

### Custom Output and Patterns

**Custom output file:**
```bash
python3 plot_fog_bar.py --out="my_analysis.png"
```

**Custom file pattern:**
```bash
python3 plot_fog_bar.py --pattern="results/fog_*.csv"
```

**Custom title:**
```bash
python3 plot_fog_bar.py --title="My Custom Analysis"
```

**Custom figure size:**
```bash
python3 plot_fog_bar.py --figsize 16 10
```

### Complete Example
```bash
python3 plot_fog_bar.py \
    --pattern="eval_output/depth_error_log_fog_*.csv" \
    --out="eval_output/depth_analysis_1sigma.png" \
    --title="Depth Error Analysis with 1σ Confidence" \
    --sigma-multiplier 1.0 \
    --figsize 16 10
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pattern` | `eval_output/depth_error_log_fog_*.csv` | Glob pattern for CSV files |
| `--out` | `eval_output/depth_error_fog_line.png` | Output PNG path |
| `--title` | `Average Absolute Depth Error and Confidence vs Fog Density` | Plot title |
| `--figsize` | `14 8` | Figure size as width height |
| `--sigma-multiplier` | `2.0` | Sigma multiplier for error bars |

## Sigma Multiplier Examples

| Multiplier | Coverage | Description |
|------------|----------|-------------|
| 0.5 | ~38% | Narrow confidence interval |
| 1.0 | ~68% | Standard deviation |
| 1.5 | ~87% | Extended confidence |
| 2.0 | ~95% | Common confidence interval |
| 2.5 | ~99% | High confidence |
| 3.0 | ~99.7% | Very high confidence |

## Output Features

### Visual Elements
- **Depth Error Line**: Blue solid line with circle markers
- **Depth Error Variance**: Blue dotted fill area
- **Confidence Line**: Green dash-dot line with square markers  
- **Confidence Variance**: Green dashed fill area
- **Grid**: Light gray grid for better readability
- **Legend**: Combined legend showing all elements

### File Output
- **Format**: PNG with 300 DPI
- **Background**: White
- **Bounding**: Tight layout with proper margins

## CSV File Requirements

The script expects CSV files with the following columns:
- `mean_abs_err_cm`: Mean absolute depth error in centimeters
- `var_abs_err_cm2`: Variance of depth error in cm²
- `conf_mean`: Mean confidence value
- `conf_var`: Variance of confidence

## Troubleshooting

### Common Issues

1. **No files found**: Check the `--pattern` parameter matches your file naming
2. **Missing data**: Ensure CSV files contain the required columns
3. **Plot not displaying**: Check if matplotlib backend is set correctly (script uses 'Agg' for headless operation)

### File Naming Convention

The script automatically parses fog density from filenames using these patterns:
- `depth_error_log_fog_0.1.csv` → fog density 0.1
- `depth_error_log_fog_1.0.csv` → fog density 1.0
- `fog_0.5_results.csv` → fog density 0.5

## Examples with Different Sigma Values

### Narrow Error Bars (0.5σ)
```bash
python3 plot_fog_bar.py --sigma-multiplier 0.5 --out="narrow_error_bars.png"
```

### Standard Error Bars (1σ)
```bash
python3 plot_fog_bar.py --sigma-multiplier 1.0 --out="standard_error_bars.png"
```

### Wide Error Bars (3σ)
```bash
python3 plot_fog_bar.py --sigma-multiplier 3.0 --out="wide_error_bars.png"
```

## Integration with Automation Script

The updated script is compatible with the `fog_experiment_automation.py` script. You can modify the automation script to use different sigma multipliers by updating the plot command in the `generate_plots()` method.

## Performance Notes

- Processing time scales with number of CSV files
- Large datasets may require increased figure size for readability
- High DPI output (300) provides publication-quality figures
