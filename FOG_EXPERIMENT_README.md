# Fog Density Experiment Automation

This directory contains scripts to automate fog density experiments for RAFT-Stereo depth estimation evaluation.

## Overview

The experiment tests how different fog densities (0.1 to 1.0) affect depth estimation accuracy and confidence. For each fog density:

1. HoloOcean simulation is started with the specified fog density
2. Depth evaluation runs for 50 seconds with ROI fraction 0.33 and sigma multiplier 1.0
3. Results are saved to CSV files named `depth_error_log_fog_X.X.csv`
4. A comparative plot is generated showing fog density vs depth error and confidence

## Files

- `fog_experiment_automation.py` - Main automation script
- `plot_fog_bar.py` - Updated plotting script that handles decimal fog densities
- `depth_eval_node.py` - Depth evaluation node (existing)
- `plot_fog_bar_original.py` - Backup of original plotting script

## Usage

### Basic Usage (Full Experiment)
```bash
python3 fog_experiment_automation.py
```

This will run experiments for fog densities: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

### Custom Fog Densities
```bash
python3 fog_experiment_automation.py --fog-densities 0.1 0.3 0.5 0.7 0.9
```

### Custom Parameters
```bash
python3 fog_experiment_automation.py \
    --recording-time 30 \
    --roi-fraction 0.5 \
    --sigma-multiplier 2.0 \
    --wait-time 10 \
    --debug
```

### Test with Fewer Densities
```bash
python3 fog_experiment_automation.py --fog-densities 0.1 0.5 1.0 --recording-time 20
```

## Parameters

- `--fog-densities`: List of fog densities to test (default: 0.1 to 1.0)
- `--recording-time`: Recording time per experiment in seconds (default: 50)
- `--roi-fraction`: ROI fraction for depth evaluation (default: 0.33 = bottom third)
- `--sigma-multiplier`: Sigma multiplier for error bars (default: 1.0)
- `--wait-time`: Wait time between experiments in seconds (default: 5)
- `--output-dir`: Output directory for results (default: ./eval_output)
- `--debug`: Enable debug mode
- `--launch-file`: Path to HoloOcean launch file
- `--depth-eval-script`: Path to depth evaluation script
- `--plot-script`: Path to plot generation script

## Output

The script generates:

1. **CSV Files**: `depth_error_log_fog_X.X.csv` for each fog density
2. **Plot**: `depth_error_fog_bar.png` comparing all fog densities
3. **Console Output**: Progress and summary information

## Prerequisites

- ROS environment must be sourced
- HoloOcean ROS package must be available
- RAFT-Stereo must be built and available
- Python 3 with required packages (numpy, matplotlib, etc.)

## Example Output

```
Starting Fog Density Experiment Automation
Testing fog densities: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Recording time per experiment: 50 seconds
Output directory: /home/madhushree/RAFT-Stereo/eval_output

============================================================
Running experiment with fog density: 0.1
============================================================
Starting HoloOcean with fog density: 0.1
Waiting for HoloOcean to initialize...
HoloOcean started successfully
Starting depth evaluation for fog density: 0.1
Recording depth data for 50 seconds...
Depth evaluation completed
Renamed depth_error_log.csv to depth_error_log_fog_0.1.csv
Cleaning up processes...
Waiting 5 seconds before next experiment...
✓ Experiment with fog density 0.1 completed successfully

...

============================================================
EXPERIMENT SUMMARY
============================================================
Successful experiments: 10
Failed experiments: 0
Successful fog densities: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Results saved in: /home/madhushree/RAFT-Stereo/eval_output
```

## Troubleshooting

1. **ROS Environment**: Make sure ROS is sourced (`source /opt/ros/noetic/setup.bash`)
2. **HoloOcean**: Ensure HoloOcean ROS package is available and launch file exists
3. **RAFT-Stereo**: Verify RAFT-Stereo is built and depth_eval_node.py is executable
4. **Permissions**: Make sure scripts are executable (`chmod +x *.py`)
5. **Dependencies**: Install required Python packages (`pip3 install numpy matplotlib`)

## Manual Plot Generation

To generate plots from existing CSV files:

```bash
python3 plot_fog_bar.py --pattern="eval_output/depth_error_log_fog_*.csv" --out="eval_output/custom_plot.png"
```
