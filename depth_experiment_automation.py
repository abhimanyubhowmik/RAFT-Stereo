#!/usr/bin/env python3
"""
Depth Experiment Automation Script

This script automates running depth evaluation across different depths by:
1. Editing the HoloOcean world JSON to set the agent z-location
2. Launching HoloOcean headless
3. Running the depth evaluation node for a fixed duration
4. Renaming the output CSV to include the tested depth
5. Generating a plot of depth vs avg absolute depth error (and confidence)

Usage:
    python3 depth_experiment_automation.py [options]
"""

import os
import sys
import time
import subprocess
import signal
import argparse
import shutil
import json
from pathlib import Path


class DepthExperimentAutomation:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Depth values to test (z-values)
        self.depth_values = args.depth_values

        # Processes
        self.holoocean_process = None
        self.depth_eval_process = None

        # Paths
        self.world_json_path = Path(args.world_json_path)
        self.launch_file = args.launch_file
        self.depth_eval_script = args.depth_eval_script
        self.plot_script = args.plot_script

        # Backup for world JSON content to restore after experiments
        self._world_json_backup = None

    def setup_signal_handlers(self):
        def handler(signum, frame):
            print("\nReceived signal {}. Shutting down gracefully...".format(signum))
            self.cleanup()
            self.restore_world_json()
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    # ---------------------- World JSON helpers ----------------------
    def read_world_json(self):
        with open(self.world_json_path, 'r') as f:
            return json.load(f)

    def write_world_json(self, data):
        # Write atomically
        tmp_path = self.world_json_path.with_suffix(self.world_json_path.suffix + '.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, self.world_json_path)

    def backup_world_json(self):
        if self._world_json_backup is None:
            try:
                with open(self.world_json_path, 'r') as f:
                    self._world_json_backup = f.read()
            except Exception as e:
                print("Warning: failed to backup world JSON: {}".format(e))

    def restore_world_json(self):
        if self._world_json_backup is not None:
            try:
                with open(self.world_json_path, 'w') as f:
                    f.write(self._world_json_backup)
                print("Restored original world JSON")
            except Exception as e:
                print("Warning: failed to restore world JSON: {}".format(e))

    def set_world_depth(self, depth_z):
        try:
            data = self.read_world_json()
            # Assume first agent controls the camera rig
            if 'agents' in data and isinstance(data['agents'], list) and data['agents']:
                loc = data['agents'][0].get('location', [0, 0, depth_z])
                if not isinstance(loc, list) or len(loc) < 3:
                    loc = [0, 0, depth_z]
                loc[2] = float(depth_z)
                data['agents'][0]['location'] = loc
                self.write_world_json(data)
                return True
            else:
                print("Error: Could not find 'agents' in world JSON to set depth")
                return False
        except Exception as e:
            print("Error setting world depth: {}".format(e))
            return False

    # ---------------------- Process management ----------------------
    def cleanup(self):
        print("Cleaning up processes...")
        if self.depth_eval_process:
            try:
                self.depth_eval_process.terminate()
                self.depth_eval_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.depth_eval_process.kill()
            except Exception as e:
                print("Error terminating depth_eval_process: {}".format(e))
            finally:
                self.depth_eval_process = None
        if self.holoocean_process:
            try:
                self.holoocean_process.terminate()
                self.holoocean_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.holoocean_process.kill()
            except Exception as e:
                print("Error terminating holoocean_process: {}".format(e))
            finally:
                self.holoocean_process = None

    def start_holoocean(self, fog_density=None):
        print("Starting HoloOcean (fog_density={} for compatibility)".format(fog_density))
        cmd = [
            'roslaunch',
            'holoocean_ros',
            'fog_holoocean.launch',
            'fog_density:={}'.format(fog_density if fog_density is not None else 0.0),
            'show_viewport:=false'
        ]
        try:
            self.holoocean_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            print("Waiting for HoloOcean to initialize...")
            time.sleep(self.args.holoocean_boot_wait)
            if self.holoocean_process.poll() is not None:
                stdout, stderr = self.holoocean_process.communicate()
                print("HoloOcean failed to start. stdout:\n{}".format(stdout.decode(errors='ignore')))
                print("stderr:\n{}".format(stderr.decode(errors='ignore')))
                return False
            print("HoloOcean started successfully")
            return True
        except Exception as e:
            print("Error starting HoloOcean: {}".format(e))
            return False

    def run_depth_eval(self, depth_z):
        print("Starting depth evaluation for depth: {}".format(depth_z))
        cmd = [
            'python3',
            str(self.depth_eval_script),
            '_roi_fraction:={}'.format(self.args.roi_fraction),
            '_sigma_multiplier:={}'.format(self.args.sigma_multiplier),
            '_total_recording_time:={}'.format(self.args.recording_time),
            '_output_dir:={}'.format(self.output_dir),
            '_debug:={}'.format(self.args.debug)
        ]
        try:
            self.depth_eval_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            print("Recording depth data for {} seconds...".format(self.args.recording_time))
            time.sleep(self.args.recording_time + 5)
            if self.depth_eval_process.poll() is None:
                print("Depth evaluation still running, terminating...")
                self.depth_eval_process.terminate()
                try:
                    self.depth_eval_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.depth_eval_process.kill()
            stdout, stderr = self.depth_eval_process.communicate()
            if stderr:
                print("Depth evaluation stderr:\n{}".format(stderr.decode(errors='ignore')))
            print("Depth evaluation completed")
            return True
        except Exception as e:
            print("Error running depth evaluation: {}".format(e))
            return False

    def rename_latest_csv(self, depth_z):
        # depth_eval_node writes a fixed file name; rename to include depth
        src = self.output_dir / 'depth_error_log.csv'
        if not src.exists():
            print("No CSV found to rename for depth {}".format(depth_z))
            return False
        dst = self.output_dir / 'depth_error_log_depth_{:.1f}.csv'.format(float(depth_z))
        try:
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            print("Renamed {} -> {}".format(src.name, dst.name))
            return True
        except Exception as e:
            print("Error renaming CSV: {}".format(e))
            return False

    def run_single_experiment(self, depth_z):
        print("\n" + "=" * 60)
        print("Running experiment at depth (z): {}".format(depth_z))
        print("=" * 60)

        # Update world depth
        if not self.set_world_depth(depth_z):
            return False

        # Start HoloOcean
        if not self.start_holoocean(fog_density=self.args.fog_density):
            return False

        # Run eval
        ok = self.run_depth_eval(depth_z)

        # Rename output
        if ok:
            self.rename_latest_csv(depth_z)

        # Cleanup sim
        self.cleanup()

        # Wait between experiments
        print("Waiting {} seconds before next experiment...".format(self.args.wait_time))
        time.sleep(self.args.wait_time)

        return ok

    def generate_plots(self):
        print("\nGenerating depth plots...")
        plot_cmd = [
            'python3',
            str(self.plot_script),
            '--pattern={}/depth_error_log_depth_*.csv'.format(self.output_dir),
            '--out={}/depth_error_depth_line.png'.format(self.output_dir),
            '--title=Average Absolute Depth Error and Confidence vs Depth',
            '--sigma-multiplier={}'.format(self.args.sigma_multiplier)
        ]
        try:
            result = subprocess.run(plot_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("Plots generated successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print("Error generating plots: {}".format(result.stderr))
                return False
        except Exception as e:
            print("Error running plot script: {}".format(e))
            return False

    def run_experiment(self):
        print("Starting Depth Experiment Automation")
        print("Depths to test: {}".format(self.depth_values))
        print("Recording time per experiment: {} seconds".format(self.args.recording_time))
        print("Output directory: {}".format(self.output_dir))

        self.setup_signal_handlers()
        self.backup_world_json()

        successes, failures = [], []
        try:
            for depth_z in self.depth_values:
                try:
                    if self.run_single_experiment(depth_z):
                        successes.append(depth_z)
                        print("\u2713 Experiment at depth {} completed successfully".format(depth_z))
                    else:
                        failures.append(depth_z)
                        print("\u2717 Experiment at depth {} failed".format(depth_z))
                except KeyboardInterrupt:
                    print("\nExperiment interrupted by user")
                    break
                except Exception as e:
                    print("\u2717 Unexpected error at depth {}: {}".format(depth_z, e))
                    failures.append(depth_z)
        finally:
            # Restore world JSON regardless of outcome
            self.restore_world_json()

        # Plot results if any success
        if successes:
            print("\nGenerating plots for {} successful experiments...".format(len(successes)))
            self.generate_plots()

        # Summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print("Successful experiments: {}".format(len(successes)))
        print("Failed experiments: {}".format(len(failures)))
        if successes:
            print("Successful depths: {}".format(successes))
        if failures:
            print("Failed depths: {}".format(failures))
        print("Results saved in: {}".format(self.output_dir))

        return len(failures) == 0


def main():
    parser = argparse.ArgumentParser(description='Automate depth experiments by editing HoloOcean world JSON and running evaluation.')

    parser.add_argument('--world-json-path',
                        default='/home/madhushree/.local/share/holoocean/0.5.0/worlds/OceanSimple/OceanSimple-Hovering2CameraOnly-VisibilityEstimation.json',
                        help='Path to HoloOcean world JSON file to edit (z-location)')

    parser.add_argument('--launch-file',
                        default='/home/madhushree/simulation_ws/src/holoocean_ros/launch/fog_holoocean.launch',
                        help='Path to HoloOcean ROS launch file (kept for compatibility)')

    parser.add_argument('--depth-eval-script',
                        default='/home/madhushree/RAFT-Stereo/depth_eval_node.py',
                        help='Path to depth evaluation script')

    parser.add_argument('--plot-script',
                        default='/home/madhushree/RAFT-Stereo/plot_depth_bar.py',
                        help='Path to plot generation script for depth experiments')

    parser.add_argument('--output-dir',
                        default='/home/madhushree/RAFT-Stereo/eval_output',
                        help='Output directory for results')

    parser.add_argument('--roi-fraction', type=float, default=0.33,
                        help='ROI fraction for depth evaluation (0.33 = bottom third)')

    parser.add_argument('--sigma-multiplier', type=float, default=1.0,
                        help='Sigma multiplier for error bars')

    parser.add_argument('--recording-time', type=int, default=50,
                        help='Recording time per experiment in seconds')

    parser.add_argument('--wait-time', type=int, default=5,
                        help='Wait time between experiments in seconds')

    parser.add_argument('--fog-density', type=float, default=0.0,
                        help='Fog density to use while testing depth (compatibility with launcher)')

    parser.add_argument('--holoocean-boot-wait', type=int, default=10,
                        help='Seconds to wait for HoloOcean to initialize after launch')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--depth-values', nargs='+', type=float,
                        default=[7, 9, 11, 13, 15, 17, 19],
                        help='List of z-depths to test')

    args = parser.parse_args()

    automation = DepthExperimentAutomation(args)
    success = automation.run_experiment()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 