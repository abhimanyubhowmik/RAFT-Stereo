#!/usr/bin/env python3
"""
Depth+Fog Experiment Automation Script

For each fog density, runs depth evaluation at multiple depths by:
1. Editing the HoloOcean world JSON z-location for each depth
2. Launching HoloOcean headless with the fog density
3. Capturing median depth from /camera/depth/image_raw topic
4. Running the depth evaluation for a fixed duration
5. Renaming output CSVs to include both fog and captured median depth
6. Generating a multi-series plot (one colored line per fog density)
7. Capturing and visualizing reference images for different depth and fog combinations

Usage:
    python3 depth_fog_experiment_automation.py [options]
"""

import os
import sys
import time
import subprocess
import signal
import argparse
import shutil
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class DepthCapture:
    """Helper class to capture and calculate median depth from ROS topic"""
    
    def __init__(self, topic='/camera/depth/image_raw', timeout=10.0):
        self.topic = topic
        self.timeout = timeout
        self.bridge = CvBridge()
        self.depth_data = None
        self.received = False
        
    def depth_callback(self, msg):
        """Callback to process depth image"""
        try:
            # Convert ROS Image message to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_array = np.asarray(depth_image, dtype=np.float32)
            
            # Filter out invalid depths (NaN, inf, negative values)
            valid_depths = depth_array[np.isfinite(depth_array) & (depth_array > 0.0)]
            
            if len(valid_depths) > 0:
                self.depth_data = np.median(valid_depths)
                self.received = True
                rospy.loginfo("Captured median depth: {:.2f}m".format(self.depth_data))
            else:
                rospy.logwarn("No valid depth values found in image")
                
        except Exception as e:
            rospy.logerr("Error processing depth image: {}".format(e))
    
    def capture_median_depth(self):
        """Capture a single depth image and return median depth value"""
        self.depth_data = None
        self.received = False
        
        # Subscribe to depth topic
        sub = rospy.Subscriber(self.topic, Image, self.depth_callback)
        
        start_time = time.time()
        rospy.loginfo("Waiting for depth data from topic: {}".format(self.topic))
        
        # Wait for data with timeout
        while not self.received and (time.time() - start_time) < self.timeout:
            time.sleep(0.1)
            
        sub.unregister()
        
        if self.received and self.depth_data is not None:
            rospy.loginfo("Successfully captured median depth: {:.2f}m".format(self.depth_data))
            return self.depth_data
        else:
            rospy.logwarn("Failed to capture depth data within {} seconds".format(self.timeout))
            return None


class DepthFogImageVisualizer:
    """Helper class for capturing and visualizing images across different depth and fog combinations"""
    
    def __init__(self, output_dir, capture_timeout=5.0):
        self.output_dir = Path(output_dir)
        self.capture_timeout = capture_timeout
        self.bridge = CvBridge()
        self.samples = []
        
    def capture_images(self, fog_density, depth_z, captured_depth, holoocean_process):
        """Capture synchronized images for a given fog density and depth"""
        print("Capturing reference images for fog {} depth z={} (captured: {:.2f}m)...".format(
            fog_density, depth_z, captured_depth))
        
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
            print("Warning: No synchronized sample received for fog {} depth {} within {}s".format(
                fog_density, depth_z, self.capture_timeout))
            return None

        cam_msg, gt_msg, est_msg, conf_msg = got_sample['data']

        # Convert messages using CvBridge
        cam_img = self._to_bgr(cam_msg)
        gt_depth = self._to_float32(gt_msg)
        est_depth = self._to_float32(est_msg)
        conf_img = self._to_float32(conf_msg)

        return {
            'fog': float(fog_density),
            'depth_z': float(depth_z),
            'captured_depth': float(captured_depth),
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

    def generate_visualization(self, title="Depth+Fog Visualization"):
        """Generate and save the visualization grid"""
        if not self.samples:
            print("No samples to visualize")
            return False

        print("Generating depth+fog visualization...")

        # Compute common display ranges
        gt_vals = np.concatenate([s['gt'].flatten() for s in self.samples])
        est_vals = np.concatenate([s['est'].flatten() for s in self.samples])
        gt_min, gt_max = np.nanpercentile(gt_vals, [2, 98]) if np.isfinite(gt_vals).any() else (0.0, 10.0)
        est_min, est_max = np.nanpercentile(est_vals, [2, 98]) if np.isfinite(est_vals).any() else (0.0, 10.0)
        conf_min, conf_max = 0.0, 1.0

        # Render grid - organize by fog density, then by depth
        samples_by_fog = {}
        for sample in self.samples:
            fog = sample['fog']
            if fog not in samples_by_fog:
                samples_by_fog[fog] = []
            samples_by_fog[fog].append(sample)

        # Create subplot for each fog density
        fog_densities = sorted(samples_by_fog.keys())
        n_fogs = len(fog_densities)
        
        if n_fogs == 0:
            print("No samples to visualize")
            return False

        # Find max number of depths for any fog density
        max_depths = max(len(samples_by_fog[fog]) for fog in fog_densities)
        
        fig, axes = plt.subplots(n_fogs, max_depths * 4, figsize=(4*max_depths*4, 3*n_fogs))
        if n_fogs == 1:
            axes = axes.reshape(1, -1)
        if max_depths == 1:
            axes = axes.reshape(-1, 4)

        for fog_idx, fog in enumerate(fog_densities):
            samples = sorted(samples_by_fog[fog], key=lambda x: x['captured_depth'])
            
            for depth_idx, sample in enumerate(samples):
                col_start = depth_idx * 4
                
                # RGB Camera (enhanced)
                ax = axes[fog_idx, col_start]
                rgb = sample.get('rgb', sample.get('cam'))
                ax.imshow(self._enhance_rgb(rgb))
                ax.set_title('Camera\nfog {:.1f}, depth {:.1f}m'.format(sample['fog'], sample['captured_depth']), fontsize=8)
                ax.axis('off')
                
                # GT Depth
                ax = axes[fog_idx, col_start + 1]
                im1 = ax.imshow(self._apply_cmap(sample['gt'], vmin=gt_min, vmax=gt_max), cmap='turbo')
                ax.set_title('GT Depth', fontsize=8)
                ax.axis('off')
                
                # Est Depth
                ax = axes[fog_idx, col_start + 2]
                im2 = ax.imshow(self._apply_cmap(sample['est'], vmin=est_min, vmax=est_max), cmap='turbo')
                ax.set_title('Est Depth', fontsize=8)
                ax.axis('off')
                
                # Confidence
                ax = axes[fog_idx, col_start + 3]
                im3 = ax.imshow(self._apply_cmap(sample['conf'], vmin=conf_min, vmax=conf_max), cmap='viridis')
                ax.set_title('Confidence', fontsize=8)
                ax.axis('off')
            
            # Row label
            axes[fog_idx, 0].set_ylabel('fog {:.1f}'.format(fog), rotation=90, fontsize=10)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        output_path = self.output_dir / 'depth_fog_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('Saved depth+fog visualization to {}'.format(output_path))
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


class DepthFogExperimentAutomation:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.fog_densities = args.fog_densities
        self.depth_values = args.depth_values

        self.holoocean_process = None
        self.depth_eval_process = None

        self.world_json_path = Path(args.world_json_path)
        self.depth_eval_script = args.depth_eval_script
        self.plot_script = args.plot_script

        self._world_json_backup = None
        
        # Initialize image visualizer
        self.image_visualizer = DepthFogImageVisualizer(self.output_dir, args.image_capture_timeout)
        
        # Initialize ROS node for depth capture and image visualization
        try:
            rospy.init_node('depth_fog_automation', anonymous=True)
            self.depth_capture = DepthCapture(topic=args.depth_topic, timeout=args.depth_capture_timeout)
            rospy.loginfo("ROS node initialized for depth capture and image visualization")
        except Exception as e:
            print("Warning: Failed to initialize ROS node: {}. Depth capture and image visualization may not work.".format(e))
            self.depth_capture = None

    # -------------------------- Signals ---------------------------
    def setup_signal_handlers(self):
        def handler(signum, frame):
            print("\nReceived signal {}. Shutting down...".format(signum))
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
            if 'agents' in data and isinstance(data['agents'], list) and data['agents']:
                loc = data['agents'][0].get('location', [0, 0, depth_z])
                if not isinstance(loc, list) or len(loc) < 3:
                    loc = [0, 0, depth_z]
                loc[2] = float(depth_z)
                data['agents'][0]['location'] = loc
                self.write_world_json(data)
                return True
            else:
                print("Error: Could not find 'agents' in world JSON")
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

    def start_holoocean(self, fog_density):
        print("Starting HoloOcean with fog_density={}".format(fog_density))
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

    def run_depth_eval(self):
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
            return True
        except Exception as e:
            print("Error running depth evaluation: {}".format(e))
            return False

    def capture_scene_depth(self):
        """Capture the median depth of the current scene"""
        if self.depth_capture is None:
            print("Warning: Depth capture not available, using fallback depth value")
            return 10.0  # fallback value
            
        try:
            # Wait a bit for the scene to stabilize
            time.sleep(2.0)
            median_depth = self.depth_capture.capture_median_depth()
            if median_depth is not None:
                return float(median_depth)
            else:
                print("Warning: Failed to capture depth, using fallback value")
                return 10.0  # fallback value
        except Exception as e:
            print("Error capturing scene depth: {}. Using fallback value.".format(e))
            return 10.0  # fallback value

    def rename_csv(self, fog_density, captured_depth):
        """Rename CSV file using captured median depth instead of depth_z"""
        src = self.output_dir / 'depth_error_log.csv'
        if not src.exists():
            print("No CSV found to rename for fog {} depth {}".format(fog_density, captured_depth))
            return False
        dst = self.output_dir / 'depth_error_log_fog_{:.1f}_depth_{:.1f}.csv'.format(
            float(fog_density), float(captured_depth))
        try:
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            print("Saved {}".format(dst.name))
            return True
        except Exception as e:
            print("Error renaming CSV: {}".format(e))
            return False

    # ------------------------- Orchestration ------------------------
    def run_single_depth(self, fog_density, depth_z):
        """Run experiment for a single depth configuration"""
        print("- Setting world depth z={} for fog {}".format(depth_z, fog_density))
        if not self.set_world_depth(depth_z):
            return False, None
            
        if not self.start_holoocean(fog_density):
            return False, None
            
        # Capture the actual median depth from the scene
        captured_depth = self.capture_scene_depth()
        print("- Captured median scene depth: {:.2f}m".format(captured_depth))
        
        # Capture reference images if enabled
        if self.args.capture_images:
            sample = self.image_visualizer.capture_images(fog_density, depth_z, captured_depth, self.holoocean_process)
            self.image_visualizer.add_sample(sample)
        
        ok = self.run_depth_eval()
        if ok:
            self.rename_csv(fog_density, captured_depth)
        self.cleanup()
        time.sleep(self.args.wait_time)
        return ok, captured_depth

    def run_experiment(self):
        """Run the complete experiment across all fog densities and depths"""
        print("Starting Depth+Fog Experiment Automation")
        print("Fog densities: {}".format(self.fog_densities))
        print("Depths: {}".format(self.depth_values))
        print("Output directory: {}".format(self.output_dir))
        if self.args.capture_images:
            print("Image capture enabled")

        self.setup_signal_handlers()
        self.backup_world_json()

        overall_success = True
        captured_depths = []  # Store all captured depths for reference
        
        try:
            for fog in self.fog_densities:
                print("\n" + "="*60)
                print("Running fog density: {}".format(fog))
                print("="*60)
                for depth in self.depth_values:
                    try:
                        success, captured_depth = self.run_single_depth(fog, depth)
                        if not success:
                            overall_success = False
                            print("\u2717 Failed at fog {}, depth z={} (captured: {:.2f}m)".format(
                                fog, depth, captured_depth if captured_depth else 0.0))
                        else:
                            captured_depths.append((fog, depth, captured_depth))
                            print("\u2713 Completed fog {}, depth z={} (captured: {:.2f}m)".format(
                                fog, depth, captured_depth))
                    except KeyboardInterrupt:
                        print("\nInterrupted by user")
                        overall_success = False
                        return overall_success
                    except Exception as e:
                        print("\u2717 Unexpected error at fog {}, depth {}: {}".format(fog, depth, e))
                        overall_success = False
        finally:
            self.restore_world_json()

        # Print summary of captured depths
        if captured_depths:
            print("\nSummary of captured depths:")
            for fog, set_depth, captured_depth in captured_depths:
                print("  Fog {:.1f}: set z={:.1f}m -> captured median={:.2f}m".format(
                    fog, set_depth, captured_depth))

        # Generate combined plot
        self.generate_plots()
        
        # Generate image visualization
        if self.args.capture_images and self.image_visualizer.samples:
            print("\nGenerating depth+fog visualization...")
            self.image_visualizer.generate_visualization("Depth+Fog Visualization")
        
        return overall_success

    def generate_plots(self):
        print("\nGenerating multi-series depth plot...")
        plot_cmd = [
            'python3',
            str(self.plot_script),
            '--pattern={}/depth_error_log_fog_*_depth_*.csv'.format(self.output_dir),
            '--out={}/depth_error_depth_multi.png'.format(self.output_dir),
            '--title=Depth Error vs Depth (colored by fog density)',
            '--sigma-multiplier={}'.format(self.args.sigma_multiplier)
        ]
        try:
            result = subprocess.run(plot_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("Plot generated: {}".format(self.output_dir / 'depth_error_depth_multi.png'))
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print("Error generating plot: {}".format(result.stderr))
                return False
        except Exception as e:
            print("Error running plot script: {}".format(e))
            return False


def main():
    parser = argparse.ArgumentParser(description='Automate depth experiments across multiple fog densities and depths.')

    parser.add_argument('--world-json-path',
                        default='/home/madhushree/.local/share/holoocean/0.5.0/worlds/OceanSimple/OceanSimple-Hovering2CameraOnly-VisibilityEstimation.json',
                        help='Path to HoloOcean world JSON file to edit (z-location)')

    parser.add_argument('--depth-eval-script',
                        default='/home/madhushree/RAFT-Stereo/depth_eval_node.py',
                        help='Path to depth evaluation script')

    parser.add_argument('--plot-script',
                        default='/home/madhushree/RAFT-Stereo/plot_depth_multi.py',
                        help='Path to multi-series depth plot script')

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

    parser.add_argument('--holoocean-boot-wait', type=int, default=10,
                        help='Seconds to wait for HoloOcean to initialize after launch')

    # New arguments for depth capture
    parser.add_argument('--depth-topic', default='/camera/depth/image_raw',
                        help='ROS topic for depth images')
    
    parser.add_argument('--depth-capture-timeout', type=float, default=10.0,
                        help='Timeout in seconds for capturing depth data')

    # Image capture arguments
    parser.add_argument('--capture-images',
                       action='store_true',
                       help='Capture reference images for each depth and fog combination')
    
    parser.add_argument('--image-capture-timeout',
                       type=float,
                       default=5.0,
                       help='Timeout in seconds for capturing images')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--fog-densities', nargs='+', type=float,
                        default=[2.0, 3.0, 4.0, 5.0],
                        help='List of fog densities to test')

    parser.add_argument('--depth-values', nargs='+', type=float,
                        default=[5, 6, 7, 8, 9, 10],
                        help='List of z-depths to test')

    args = parser.parse_args()

    automation = DepthFogExperimentAutomation(args)
    success = automation.run_experiment()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 