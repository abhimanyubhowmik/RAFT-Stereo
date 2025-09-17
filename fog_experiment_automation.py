#!/usr/bin/env python3
"""
Fog Density Experiment Automation Script

This script automates the fog density experiment by:
1. Starting HoloOcean with different fog densities (0.1 to 1.0)
2. Running depth evaluation for 50 seconds for each fog density
3. Renaming CSV files with appropriate fog density values
4. Generating plots comparing fog density vs depth error
5. Capturing and visualizing reference images for each fog density

Usage:
    python3 fog_experiment_automation.py [options]
"""

import os
import sys
import time
import subprocess
import signal
import argparse
import shutil
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ROS imports for image visualization
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class ImageVisualizer:
    """Helper class for capturing and visualizing images across different fog densities"""
    
    def __init__(self, output_dir, capture_timeout=5.0):
        self.output_dir = Path(output_dir)
        self.capture_timeout = capture_timeout
        self.bridge = CvBridge()
        self.samples = []
        
    def capture_images(self, fog_density, holoocean_process):
        """Capture synchronized images for a given fog density"""
        print("Capturing reference images for fog density: {}".format(fog_density))
        
        # Wait for HoloOcean to stabilize
        time.sleep(2.0)
        
        # Set up message filters for approximate synchronization
        rgb_sub = Subscriber('/camera/rgb/image_raw', Image)
        gt_sub = Subscriber('/camera/depth/image_raw', Image)
        est_sub = Subscriber('/raftstereo_node/depth', Image)
        conf_sub = Subscriber('/raftstereo_node/confidence', Image)

        got_sample = {'data': None}

        def cb(rgb_msg, gt_msg, est_msg, conf_msg):
            got_sample['data'] = (rgb_msg, gt_msg, est_msg, conf_msg)

        ats = ApproximateTimeSynchronizer([rgb_sub, gt_sub, est_sub, conf_sub], queue_size=10, slop=0.2)
        ats.registerCallback(cb)

        # Wait for synchronized sample
        start_t = time.time()
        while time.time() - start_t < self.capture_timeout:
            if got_sample['data'] is not None:
                break
            rospy.sleep(0.05)

        if got_sample['data'] is None:
            print("Warning: No synchronized sample received for fog {} within {}s".format(
                fog_density, self.capture_timeout))
            return None

        cam_msg, gt_msg, est_msg, conf_msg = got_sample['data']

        # Convert messages using CvBridge
        cam_img = self._to_bgr(cam_msg)
        gt_depth = self._to_float32(gt_msg)
        est_depth = self._to_float32(est_msg)
        conf_img = self._to_float32(conf_msg)

        return {
            'fog': float(fog_density),
            'rgb': cam_img,
            'gt': gt_depth,
            'est': est_depth,
            'conf': conf_img,
        }

    def _to_bgr(self, img_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            return cv_img
        except Exception:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            return cv_img

    def _to_float32(self, img_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            cv_img = np.asarray(cv_img)
            if cv_img.dtype != np.float32:
                cv_img = cv_img.astype(np.float32)
            return cv_img
        except Exception:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
            cv_img = (cv_img.astype(np.float32) / 255.0)
            return cv_img

    def add_sample(self, sample):
        """Add a captured sample to the collection"""
        if sample is not None:
            self.samples.append(sample)

    def generate_visualization(self, title="Fog Density Visualization"):
        """Generate and save the visualization grid"""
        if not self.samples:
            print("No samples to visualize")
            return False

        print("Generating fog density visualization...")

        # Compute common display ranges
        gt_vals = np.concatenate([s['gt'].flatten() for s in self.samples])
        est_vals = np.concatenate([s['est'].flatten() for s in self.samples])
        gt_min, gt_max = np.nanpercentile(gt_vals, [2, 98]) if np.isfinite(gt_vals).any() else (0.0, 10.0)
        est_min, est_max = np.nanpercentile(est_vals, [2, 98]) if np.isfinite(est_vals).any() else (0.0, 10.0)
        conf_min, conf_max = 0.0, 1.0

        # Render grid
        rows = len(self.samples)
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = np.array([axes])

        for r, s in enumerate(sorted(self.samples, key=lambda d: d['fog'])):
            # RGB Camera (enhanced)
            ax = axes[r, 0]
            rgb = s.get('rgb', s.get('cam'))
            ax.imshow(self._enhance_rgb(rgb))
            ax.set_title('Camera', fontsize=10)
            ax.axis('off')
            # GT Depth
            ax = axes[r, 1]
            im1 = ax.imshow(self._apply_cmap(s['gt'], vmin=gt_min, vmax=gt_max), cmap='turbo')
            ax.set_title('GT Depth', fontsize=10)
            ax.axis('off')
            # Est Depth
            ax = axes[r, 2]
            im2 = ax.imshow(self._apply_cmap(s['est'], vmin=est_min, vmax=est_max), cmap='turbo')
            ax.set_title('Est Depth', fontsize=10)
            ax.axis('off')
            # Confidence
            ax = axes[r, 3]
            im3 = ax.imshow(self._apply_cmap(s['conf'], vmin=conf_min, vmax=conf_max), cmap='viridis')
            ax.set_title('Confidence', fontsize=10)
            ax.axis('off')
            # Row label
            axes[r, 0].set_ylabel('fog {:.1f}'.format(s['fog']), rotation=90, fontsize=10)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        output_path = self.output_dir / 'fog_density_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('Saved fog density visualization to {}'.format(output_path))
        return True

    def _apply_cmap(self, img, vmin=None, vmax=None):
        """Normalize to 0..1 based on vmin/vmax, handle NaNs"""
        a = np.array(img, dtype=np.float32)
        if vmin is None:
            vmin = np.nanmin(a)
        if vmax is None:
            vmax = np.nanmax(a)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        a = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
        return a

    def _percentile_minmax(self, img, pmin=2.0, pmax=98.0):
        a = np.asarray(img, dtype=np.float32)
        if a.size == 0:
            return 0.0, 1.0
        vmin = float(np.nanpercentile(a, pmin))
        vmax = float(np.nanpercentile(a, pmax))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def _enhance_rgb(self, rgb_img, pmin=2.0, pmax=98.0):
        """Enhance RGB image by scaling brightness based on luminance percentiles"""
        img = np.asarray(rgb_img)
        if img.ndim == 2:
            vmin, vmax = self._percentile_minmax(img, pmin, pmax)
            g = self._apply_cmap(img, vmin=vmin, vmax=vmax)
            return np.stack([g, g, g], axis=-1)
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Y = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
        vmin, vmax = self._percentile_minmax(Y, pmin, pmax)
        if vmax <= vmin:
            vmax = vmin + 1.0
        scale = 1.0 / (vmax - vmin)
        offset = -vmin * scale
        rgb_enh = np.clip(rgb * scale + offset, 0.0, 1.0)
        if np.nanmax(rgb) > 1.5:
            rgb_enh = np.clip((rgb - vmin) / (vmax - vmin), 0.0, 1.0)
        return rgb_enh


class FogExperimentAutomation:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Fog density values to test
        self.fog_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Process handles
        self.holoocean_process = None
        self.depth_eval_process = None
        
        # Paths
        self.launch_file = args.launch_file
        self.depth_eval_script = args.depth_eval_script
        self.plot_script = args.plot_script
        
        # Initialize image visualizer
        self.image_visualizer = ImageVisualizer(self.output_dir, args.image_capture_timeout)
        
        # Initialize ROS node for image capture
        try:
            rospy.init_node('fog_experiment_automation', anonymous=True)
            print("ROS node initialized for image capture")
        except Exception as e:
            print("Warning: Failed to initialize ROS node: {}. Image capture may not work.".format(e))
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\nReceived signal {}. Shutting down gracefully...".format(signum))
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def cleanup(self):
        """Clean up running processes"""
        print("Cleaning up processes...")
        
        if self.depth_eval_process:
            try:
                self.depth_eval_process.terminate()
                self.depth_eval_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.depth_eval_process.kill()
            except Exception as e:
                print("Error terminating depth_eval_process: {}".format(e))
        
        if self.holoocean_process:
            try:
                self.holoocean_process.terminate()
                self.holoocean_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.holoocean_process.kill()
            except Exception as e:
                print("Error terminating holoocean_process: {}".format(e))
    
    def start_holoocean(self, fog_density):
        """Start HoloOcean with specified fog density"""
        print("Starting HoloOcean with fog density: {}".format(fog_density))
        
        cmd = [
            'roslaunch', 
            'holoocean_ros', 
            'fog_holoocean.launch',
            'fog_density:={}'.format(fog_density),
            'fog_density_low:={}'.format(fog_density),
            'fog_density_high:={}'.format(fog_density),
            'show_viewport:=false'
        ]
        
        try:
            self.holoocean_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait a bit for HoloOcean to initialize
            print("Waiting for HoloOcean to initialize...")
            time.sleep(10)
            
            if self.holoocean_process.poll() is not None:
                stdout, stderr = self.holoocean_process.communicate()
                print("HoloOcean failed to start. stdout: {}".format(stdout.decode()))
                print("stderr: {}".format(stderr.decode()))
                return False
            
            print("HoloOcean started successfully")
            return True
            
        except Exception as e:
            print("Error starting HoloOcean: {}".format(e))
            return False
    
    def start_depth_evaluation(self, fog_density):
        """Start depth evaluation node"""
        print("Starting depth evaluation for fog density: {}".format(fog_density))
        
        cmd = [
            'python3', 
            'depth_eval_node.py',
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
            
            # Wait for the recording to complete
            print("Recording depth data for {} seconds...".format(self.args.recording_time))
            time.sleep(self.args.recording_time + 5)  # Extra buffer time
            
            # Check if process completed successfully
            if self.depth_eval_process.poll() is None:
                print("Depth evaluation still running, terminating...")
                self.depth_eval_process.terminate()
                self.depth_eval_process.wait(timeout=5)
            
            stdout, stderr = self.depth_eval_process.communicate()
            if stderr:
                print("Depth evaluation stderr: {}".format(stderr.decode()))
            
            print("Depth evaluation completed")
            return True
            
        except Exception as e:
            print("Error running depth evaluation: {}".format(e))
            return False
    
    def rename_csv_file(self, fog_density):
        """Rename the generated CSV file with fog density"""
        csv_files = list(self.output_dir.glob("depth_error_log.csv"))
        
        if not csv_files:
            print("No CSV file found for fog density {}".format(fog_density))
            return False
        
        csv_file = max(csv_files, key=os.path.getctime)
        new_filename = "depth_error_log_fog_{:.1f}.csv".format(fog_density)
        new_path = self.output_dir / new_filename
        
        try:
            shutil.move(str(csv_file), str(new_path))
            print("Renamed {} to {}".format(csv_file.name, new_filename))
            return True
        except Exception as e:
            print("Error renaming CSV file: {}".format(e))
            return False
    
    def run_single_experiment(self, fog_density):
        """Run a single experiment with given fog density"""
        print("\n" + "="*60)
        print("Running experiment with fog density: {}".format(fog_density))
        print("="*60)
        
        # Start HoloOcean
        if not self.start_holoocean(fog_density):
            return False
        
        # Capture reference images
        if self.args.capture_images:
            sample = self.image_visualizer.capture_images(fog_density, self.holoocean_process)
            self.image_visualizer.add_sample(sample)
        
        # Start depth evaluation
        if not self.start_depth_evaluation(fog_density):
            self.cleanup()
            return False
        
        # Rename CSV file
        if not self.rename_csv_file(fog_density):
            print("Warning: Failed to rename CSV for fog density {}".format(fog_density))
        
        # Clean up processes
        self.cleanup()
        
        # Wait between experiments
        print("Waiting {} seconds before next experiment...".format(self.args.wait_time))
        time.sleep(self.args.wait_time)
        
        return True
    
    def generate_plots(self):
        """Generate plots from all collected data"""
        print("\nGenerating plots...")
        
        plot_cmd = [
            'python3', 
            str(self.plot_script),
            '--pattern={}/depth_error_log_fog_*.csv'.format(self.output_dir),
            '--out={}/depth_error_fog_bar.png'.format(self.output_dir),
            '--title=Average Absolute Depth Error and Confidence vs Fog Density'
        ]
        
        try:
            result = subprocess.run(plot_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("Plots generated successfully")
                print(result.stdout)
            else:
                print("Error generating plots: {}".format(result.stderr))
                return False
        except Exception as e:
            print("Error running plot script: {}".format(e))
            return False
        
        return True
    
    def run_experiment(self):
        """Run the complete fog density experiment"""
        print("Starting Fog Density Experiment Automation")
        print("Testing fog densities: {}".format(self.fog_densities))
        print("Recording time per experiment: {} seconds".format(self.args.recording_time))
        print("Output directory: {}".format(self.output_dir))
        if self.args.capture_images:
            print("Image capture enabled")
        
        self.setup_signal_handlers()
        
        successful_experiments = []
        failed_experiments = []
        
        for fog_density in self.fog_densities:
            try:
                if self.run_single_experiment(fog_density):
                    successful_experiments.append(fog_density)
                    print("✓ Experiment with fog density {} completed successfully".format(fog_density))
                else:
                    failed_experiments.append(fog_density)
                    print("✗ Experiment with fog density {} failed".format(fog_density))
            except KeyboardInterrupt:
                print("\nExperiment interrupted by user")
                break
            except Exception as e:
                print("✗ Unexpected error in experiment with fog density {}: {}".format(fog_density, e))
                failed_experiments.append(fog_density)
        
        # Generate plots
        if successful_experiments:
            print("\nGenerating plots for {} successful experiments...".format(len(successful_experiments)))
            self.generate_plots()
        
        # Generate image visualization
        if self.args.capture_images and self.image_visualizer.samples:
            print("\nGenerating fog density visualization...")
            self.image_visualizer.generate_visualization("Fog Density Visualization")
        
        # Summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print("Successful experiments: {}".format(len(successful_experiments)))
        print("Failed experiments: {}".format(len(failed_experiments)))
        
        if successful_experiments:
            print("Successful fog densities: {}".format(successful_experiments))
        
        if failed_experiments:
            print("Failed fog densities: {}".format(failed_experiments))
        
        print("Results saved in: {}".format(self.output_dir))
        
        return len(failed_experiments) == 0

def main():
    parser = argparse.ArgumentParser(description='Automate fog density experiments')
    
    parser.add_argument('--launch-file', 
                       default='/home/madhushree/simulation_ws/src/holoocean_ros/launch/fog_holoocean.launch',
                       help='Path to HoloOcean launch file')
    
    parser.add_argument('--depth-eval-script',
                       default='/home/madhushree/RAFT-Stereo/depth_eval_node.py',
                       help='Path to depth evaluation script')
    
    parser.add_argument('--plot-script',
                       default='/home/madhushree/RAFT-Stereo/plot_fog_bar.py',
                       help='Path to plot generation script')
    
    parser.add_argument('--output-dir',
                       default='/home/madhushree/RAFT-Stereo/eval_output',
                       help='Output directory for results')
    
    parser.add_argument('--roi-fraction',
                       type=float,
                       default=0.33,
                       help='ROI fraction for depth evaluation (0.33 = bottom third)')
    
    parser.add_argument('--sigma-multiplier',
                       type=float,
                       default=1.0,
                       help='Sigma multiplier for error bars')
    
    parser.add_argument('--recording-time',
                       type=int,
                       default=50,
                       help='Recording time per experiment in seconds')
    
    parser.add_argument('--wait-time',
                       type=int,
                       default=5,
                       help='Wait time between experiments in seconds')
    
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug mode')
    
    # Image capture arguments
    parser.add_argument('--capture-images',
                       action='store_true',
                       help='Capture reference images for each fog density')
    
    parser.add_argument('--image-capture-timeout',
                       type=float,
                       default=5.0,
                       help='Timeout in seconds for capturing images')
    
    parser.add_argument('--fog-densities',
                       nargs='+',
                       type=float,
                       default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                       help='List of fog densities to test')
    
    args = parser.parse_args()
    
    # Create experiment automation instance
    automation = FogExperimentAutomation(args)
    
    # Override fog densities if provided
    automation.fog_densities = args.fog_densities
    
    # Run the experiment
    success = automation.run_experiment()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
