#!/usr/bin/env python3
import os
import time
import math
import csv
import rospy
import numpy as np
import message_filters
from collections import deque
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DepthEvalNode:
    def __init__(self):
        self.bridge = CvBridge()

        # Params
        self.gt_topic = rospy.get_param('~gt_topic', '/camera/depth/image_raw')
        self.est_topic = rospy.get_param('~est_topic', '/raftstereo_node/depth')
        self.conf_topic = rospy.get_param('~conf_topic', '/raftstereo_node/confidence')
        self.roi_fraction = float(rospy.get_param('~roi_fraction', 0.5))  # 1.0 == full image, 0.5 == bottom half, ~0.333 == bottom third
        self.sigma_multiplier = float(rospy.get_param('~sigma_multiplier', 2.0))  # 2 or 3
        self.output_dir = rospy.get_param('~output_dir', os.path.join(rospy.get_param('~pkg_path', os.getcwd()), 'eval_output'))
        # save_interval_sec: how often to flush CSV/plot to disk during recording
        self.save_interval_sec = float(rospy.get_param('~save_interval_sec', 5.0))
        # total_recording_time: stop automatically after this many seconds (0 = run until killed)
        self.total_recording_time = float(rospy.get_param('~total_recording_time', 0.0))
        self.min_depth = float(rospy.get_param('~min_depth', 0.0))
        self.max_depth = float(rospy.get_param('~max_depth', 100.0))
        self.debug = rospy.get_param('~debug', False)

        os.makedirs(self.output_dir, exist_ok=True)

        # Buffers: keep tuples (stamp, mean_abs_err_m, var_abs_err_m2, count, density, conf_mean, conf_var)
        self.samples = deque()
        self.last_save_time = 0.0
        self.start_time = time.time()

        # Subscribers (approx sync) for GT, estimated depth, and confidence
        gt_sub = message_filters.Subscriber(self.gt_topic, Image)
        est_sub = message_filters.Subscriber(self.est_topic, Image)
        conf_sub = message_filters.Subscriber(self.conf_topic, Image)
        ats = message_filters.ApproximateTimeSynchronizer([gt_sub, est_sub, conf_sub], queue_size=20, slop=0.05)
        ats.registerCallback(self.callback)

        rospy.loginfo('DepthEvalNode started. GT=%s EST=%s CONF=%s roi_fraction=%.3f output_dir=%s total_recording_time=%.1fs',
                      self.gt_topic, self.est_topic, self.conf_topic, self.roi_fraction, self.output_dir, self.total_recording_time)

    def img_to_np32(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        arr = np.asarray(cv_img)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def compute_roi(self, *arrays):
        # arrays: each HxW
        H, W = arrays[0].shape
        frac = np.clip(self.roi_fraction, 0.0, 1.0)
        roi_h = int(round(H * frac))
        start = H - roi_h
        if start < 0:
            start = 0
        return [a[start:H, :] for a in arrays]

    def compute_abs_error_stats(self, depth_gt, depth_est):
        # Mask invalid
        valid = np.isfinite(depth_gt) & np.isfinite(depth_est) & (depth_gt > self.min_depth) & (depth_est > self.min_depth) & (depth_gt <= self.max_depth) & (depth_est <= self.max_depth)
        count = int(np.count_nonzero(valid))
        total = depth_gt.size
        density = float(count) / float(total) if total > 0 else 0.0
        if count == 0:
            return math.nan, math.nan, 0, 0.0
        abs_err = np.abs(depth_est[valid] - depth_gt[valid])  # meters
        mean_err = float(abs_err.mean())  # m
        var_err = float(abs_err.var())    # m^2
        return mean_err, var_err, count, density

    def compute_conf_stats(self, conf_roi):
        valid = np.isfinite(conf_roi)
        if not np.any(valid):
            return math.nan, math.nan
        vals = conf_roi[valid]
        return float(vals.mean()), float(vals.var())

    def save_csv_and_plot(self):
        # CSV log (all samples) in centimeters for depth, raw for confidence [unitless]
        csv_path = os.path.join(self.output_dir, 'depth_error_log.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['stamp', 'mean_abs_err_cm', 'var_abs_err_cm2', 'count', 'density', 'conf_mean', 'conf_var'])
            for s in list(self.samples):
                mean_cm = s[1] * 100.0 if np.isfinite(s[1]) else math.nan
                var_cm2 = s[2] * (100.0**2) if np.isfinite(s[2]) else math.nan
                w.writerow([f'{s[0]:.3f}', f'{mean_cm:.6f}', f'{var_cm2:.6f}', s[3], f'{s[4]:.6f}', f'{s[5]:.6f}', f'{s[6]:.6f}'])

        # Per-sample plot for depth only (confidence can be plotted separately if needed)
        stamps = np.array([s[0] for s in self.samples], dtype=np.float32)
        if stamps.size == 0:
            return

        xs = stamps - stamps[0]  # relative time in seconds
        means_cm = np.array([s[1]*100.0 if np.isfinite(s[1]) else np.nan for s in self.samples], dtype=np.float32)
        vars_cm2 = np.array([s[2]*(100.0**2) if np.isfinite(s[2]) else np.nan for s in self.samples], dtype=np.float32)
        sigma_cm = np.sqrt(np.maximum(vars_cm2, 0.0))
        k = self.sigma_multiplier

        plt.figure(figsize=(10, 5))
        plt.fill_between(xs, means_cm - k*sigma_cm, means_cm + k*sigma_cm, color='tab:blue', alpha=0.2, label=f'±{int(k)}σ (per-sample)')
        plt.plot(xs, means_cm, color='tab:blue', marker='o', markersize=2, linewidth=1.0, label='per-sample mean |ΔZ|')
        plt.title('Per-sample Absolute Depth Error (bottom ROI): mean and variance over time')
        plt.xlabel('time (s)')
        plt.ylabel('error (cm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = os.path.join(self.output_dir, 'depth_error_per_sample.png')
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

        if self.debug:
            rospy.loginfo('Saved per-sample plot and CSV (cm units) to %s', self.output_dir)

    def callback(self, gt_msg, est_msg, conf_msg):
        try:
            gt = self.img_to_np32(gt_msg)
            est = self.img_to_np32(est_msg)
            conf = self.img_to_np32(conf_msg)
            # Align sizes
            if gt.shape != est.shape:
                import cv2
                est = cv2.resize(est, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            if conf.shape != gt.shape:
                import cv2
                conf = cv2.resize(conf, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            # ROI (same spatial crop for all)
            gt_roi, est_roi, conf_roi = self.compute_roi(gt, est, conf)

            # Stats (depth in meters internally)
            mean_err, var_err, count, density = self.compute_abs_error_stats(gt_roi, est_roi)
            conf_mean, conf_var = self.compute_conf_stats(conf_roi)

            stamp = gt_msg.header.stamp.to_sec() if gt_msg.header.stamp else time.time()
            self.samples.append((stamp, mean_err, var_err, count, density, conf_mean, conf_var))

            # Periodic save during recording
            if (stamp - self.last_save_time) >= self.save_interval_sec:
                self.save_csv_and_plot()
                self.last_save_time = stamp

            # Auto-stop after total_recording_time
            if self.total_recording_time > 0.0:
                if (time.time() - self.start_time) >= self.total_recording_time:
                    # Final save before shutdown
                    self.save_csv_and_plot()
                    rospy.loginfo('Total recording time %.1fs reached. Stopping.', self.total_recording_time)
                    rospy.signal_shutdown('Recording complete')

            if self.debug:
                rospy.loginfo('Frame depth: mean=%.2f cm var=%.2f cm^2 count=%d density=%.3f | conf: mean=%.3f var=%.3f',
                              mean_err*100.0 if np.isfinite(mean_err) else float('nan'),
                              var_err*(100.0**2) if np.isfinite(var_err) else float('nan'),
                              count, density,
                              conf_mean, conf_var)
        except Exception as e:
            rospy.logerr('DepthEval error: %s', str(e))


def main():
    rospy.init_node('depth_eval_node')
    _ = DepthEvalNode()
    rospy.spin()


if __name__ == '__main__':
    main() 