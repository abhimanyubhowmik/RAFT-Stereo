#!/usr/bin/env python3
"""
Visualize images across different fog densities by capturing one synchronized sample
from these topics per fog:
- /camera/rgb/image_raw (RGB camera)
- /camera/depth/image_raw (gt depth)
- /raftstereo_node/depth (estimated depth)
- /raftstereo_node/confidence (estimated confidence)

For each fog density, the script:
1) Launches HoloOcean with the specified fog
2) Waits for the simulator to boot
3) Collects one approx-time-synchronized set of messages within a timeout window
4) Repeats for all fog densities
5) Renders a grid (rows=fog densities, cols=[camera, gt depth, est depth, confidence])

All images are normalized before plotting. RGB image is enhanced using luminance-based
percentile auto-contrast; depth/conf use percentile-based normalization across samples.

Usage:
    python3 visualize_fog_images.py [options]
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
import cv2

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class FogImageVisualizer:
    def __init__(self, args):
        self.args = args
        self.output_path = Path(args.out)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.samples = []  # list of dicts per fog: {'fog': v, 'cam': img, 'gt': depth, 'est': depth, 'conf': conf}
        self.holoocean_process = None

    def _start_holoocean(self, fog_density):
        print("Starting HoloOcean with fog_density={}".format(fog_density))
        cmd = [
            'roslaunch',
            'holoocean_ros',
            'fog_holoocean.launch',
            'fog_density:={}'.format(fog_density),
            'show_viewport:=false'
        ]
        try:
            self.holoocean_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            # Boot wait
            time.sleep(self.args.holoocean_boot_wait)
            if self.holoocean_process.poll() is not None:
                stdout, stderr = self.holoocean_process.communicate()
                print("HoloOcean failed to start. stdout:\n{}".format(stdout.decode(errors='ignore')))
                print("stderr:\n{}".format(stderr.decode(errors='ignore')))
                return False
            return True
        except Exception as e:
            print("Error starting HoloOcean: {}".format(e))
            return False

    def _stop_holoocean(self):
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

    def _capture_one_sample(self, fog_density):
        print("Capturing images for fog {}...".format(fog_density))

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

        # Wait up to capture_window seconds to receive at least one synchronized set
        start_t = time.time()
        while time.time() - start_t < self.args.capture_window:
            if got_sample['data'] is not None:
                break
            rospy.sleep(0.05)

        if got_sample['data'] is None:
            print("Warning: No synchronized sample received for fog {} within {}s".format(
                fog_density, self.args.capture_window))
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
            # Prefer passthrough to preserve bit depth; many camera topics are mono
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            return cv_img
        except Exception:
            # Fallback: try passthrough then convert if single channel
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
            # As a last resort, try to decode as mono8 and scale to [0,1]
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
            cv_img = (cv_img.astype(np.float32) / 255.0)
            return cv_img

    def run(self):
        rospy.init_node('fog_image_visualizer', anonymous=True, disable_signals=True)

        # Handle signals to shutdown cleanly
        def handler(signum, frame):
            print("\nReceived signal {}. Shutting down...".format(signum))
            self._stop_holoocean()
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        # Collect samples for each fog
        for fog in self.args.fog_densities:
            if not self._start_holoocean(fog):
                continue
            sample = self._capture_one_sample(fog)
            if sample is not None:
                self.samples.append(sample)
            self._stop_holoocean()
            time.sleep(self.args.wait_time)

        if not self.samples:
            print("No samples captured. Exiting.")
            return False

        # Compute common display ranges
        gt_vals = np.concatenate([s['gt'].flatten() for s in self.samples])
        est_vals = np.concatenate([s['est'].flatten() for s in self.samples])
        # Robust ranges via percentiles to avoid outliers
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
            ax.imshow(enhance_rgb(rgb))
            ax.set_title('Camera', fontsize=10)
            ax.axis('off')
            # GT Depth
            ax = axes[r, 1]
            im1 = ax.imshow(apply_cmap(s['gt'], vmin=gt_min, vmax=gt_max), cmap='turbo')
            ax.set_title('GT Depth', fontsize=10)
            ax.axis('off')
            # Est Depth
            ax = axes[r, 2]
            im2 = ax.imshow(apply_cmap(s['est'], vmin=est_min, vmax=est_max), cmap='turbo')
            ax.set_title('Est Depth', fontsize=10)
            ax.axis('off')
            # Confidence
            ax = axes[r, 3]
            im3 = ax.imshow(apply_cmap(s['conf'], vmin=conf_min, vmax=conf_max), cmap='viridis')
            ax.set_title('Confidence', fontsize=10)
            ax.axis('off')
            # Row label
            axes[r, 0].set_ylabel('fog {:.1f}'.format(s['fog']), rotation=90, fontsize=10)

        plt.suptitle(self.args.title, fontsize=14)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('Saved grid image to {}'.format(self.output_path))
        return True


def apply_cmap(img, vmin=None, vmax=None):
    # Normalize to 0..1 based on vmin/vmax, handle NaNs
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


def cv2rgb(bgr_img):
    # Convert BGR (OpenCV default) to RGB for matplotlib
    if bgr_img.ndim == 3 and bgr_img.shape[2] == 3:
        return bgr_img[:, :, ::-1]
    return bgr_img


def percentile_minmax(img, pmin=2.0, pmax=98.0):
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


def enhance_rgb(rgb_img, pmin=2.0, pmax=98.0):
    """Enhance RGB image by scaling brightness based on luminance percentiles.
    Preserves color balance better than per-channel scaling.
    Returns float image in [0,1].
    """
    img = np.asarray(rgb_img)
    if img.ndim == 2:
        # If mono slipped through, visualize as grayscale
        vmin, vmax = percentile_minmax(img, pmin, pmax)
        g = apply_cmap(img, vmin=vmin, vmax=vmax)
        return np.stack([g, g, g], axis=-1)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    # Convert BGR to RGB if needed (heuristic: assume already RGB from /camera/rgb/image_raw)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Luminance approximation (Rec. 601 luma with RGB order)
    Y = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    vmin, vmax = percentile_minmax(Y, pmin, pmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    # Normalize luminance to [0,1]
    scale = 1.0 / (vmax - vmin)
    offset = -vmin * scale
    rgb_enh = np.clip(rgb * scale + offset, 0.0, 1.0)
    # If original looked like 0..255, adjust scale appropriately
    if np.nanmax(rgb) > 1.5:  # values likely in 0..255
        rgb_enh = np.clip((rgb - vmin) / (vmax - vmin), 0.0, 1.0)
    return rgb_enh


def main():
    parser = argparse.ArgumentParser(description='Capture and visualize a single synchronized image set for multiple fog densities.')
    parser.add_argument('--fog-densities', nargs='+', type=float, default=[1.0, 2.0, 3.0, 4.0, 5.0],
                        help='Fog densities to visualize')
    parser.add_argument('--out', default='/home/madhushree/RAFT-Stereo/eval_output/fog_visual_grid.png',
                        help='Output image path')
    parser.add_argument('--title', default='One-sample visualization per fog density', help='Figure title')
    parser.add_argument('--holoocean-boot-wait', type=int, default=10, help='Seconds to wait after launch')
    parser.add_argument('--capture-window', type=int, default=5, help='Seconds to wait for a synchronized sample')
    parser.add_argument('--wait-time', type=int, default=3, help='Seconds to wait between fog runs')

    args = parser.parse_args()

    vis = FogImageVisualizer(args)
    ok = vis.run()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main() 