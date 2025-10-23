#!/usr/bin/env python3
import os
import sys
import rospy
import torch
import numpy as np
import message_filters
import traceback
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# Ensure core is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

from raft_stereo import RAFTStereo
from utils.utils import InputPadder


class RAFTStereoNode:
    def __init__(self):
        self.bridge = CvBridge()

        # Parameters
        self.left_topic = rospy.get_param('~left_topic', '/alphasense_driver_ros/cam1')
        self.right_topic = rospy.get_param('~right_topic', '/alphasense_driver_ros/cam0')
        self.restore_ckpt = rospy.get_param('~restore_ckpt', os.path.join(PROJECT_ROOT, 'models', 'iraftstereo_rvc.pth'))
        self.mixed_precision = rospy.get_param('~mixed_precision', False)
        self.valid_iters = rospy.get_param('~valid_iters', 32)
        self.hidden_dims = rospy.get_param('~hidden_dims', [128, 128, 128])
        self.corr_implementation = rospy.get_param('~corr_implementation', 'reg')
        # Realtime checkpoints typically use a shared backbone
        self.shared_backbone = rospy.get_param('~shared_backbone', True)
        self.corr_levels = rospy.get_param('~corr_levels', 4)
        self.corr_radius = rospy.get_param('~corr_radius', 4)
        # Realtime checkpoints often train with n_downsample=3 (factor 8)
        self.n_downsample = rospy.get_param('~n_downsample', 3)
        self.context_norm = rospy.get_param('~context_norm', 'batch')
        self.slow_fast_gru = rospy.get_param('~slow_fast_gru', False)
        # Many realtime models use 2 GRU layers
        self.n_gru_layers = rospy.get_param('~n_gru_layers', 2)
        self.device = rospy.get_param('~device', 'cuda')
        self.strict_load = rospy.get_param('~strict_load', True)
        self.compute_confidence = rospy.get_param('~compute_confidence', True)
        self.debug = rospy.get_param('~debug', False)
        self.output_frame_id = rospy.get_param('~output_frame_id', '')

        # Depth parameters
        self.fx = float(rospy.get_param('~fx', 0.0))
        self.baseline = float(rospy.get_param('~baseline', 0.0))  # in meters
        self.min_disp = float(rospy.get_param('~min_disp', 1e-3))
        self.conf_threshold = float(rospy.get_param('~conf_threshold', 0.5))
        self.publish_raw_depth = bool(rospy.get_param('~publish_raw_depth', False))
        self.max_depth = float(rospy.get_param('~max_depth', 5.5))  # maximum depth in meters
        self.depth_scale = float(rospy.get_param('~depth_scale', 1.0))  # scale factor for depth output
        # Intrinsics for point cloud
        self.fy = float(rospy.get_param('~fy', self.fx))
        self.cx = float(rospy.get_param('~cx', 0.0))
        self.cy = float(rospy.get_param('~cy', 0.0))
        self.pc_step = int(rospy.get_param('~pc_step', 1))  # subsample factor to lighten the cloud
        self.pc_use_masked_depth = bool(rospy.get_param('~pc_use_masked_depth', True))
        # YAML path for intrinsics (left camera). Optionally ~right_camera_yaml to infer baseline
        self.camera_yaml = rospy.get_param('~left_camera_yaml', '')
        self.right_camera_yaml = rospy.get_param('~right_camera_yaml', '')
        Tx_left = None
        Tx_right = None
        if self.camera_yaml:
            try:
                with open(self.camera_yaml, 'r') as f:
                    cam = yaml.safe_load(f)
                K = cam.get('camera_matrix', {}).get('data', None)
                P = cam.get('projection_matrix', {}).get('data', None)
                if K and len(K) == 9:
                    self.fx = float(K[0]); self.fy = float(K[4]); self.cx = float(K[2]); self.cy = float(K[5])
                if P and len(P) == 12:
                    Tx_left = float(P[3])
                rospy.loginfo('Loaded left intrinsics from %s: fx=%.3f fy=%.3f cx=%.3f cy=%.3f Tx_left=%s', self.camera_yaml, self.fx, self.fy, self.cx, self.cy, str(Tx_left))
            except Exception as e:
                rospy.logwarn('Failed to load camera intrinsics from %s: %s', self.camera_yaml, str(e))
        if self.right_camera_yaml:
            try:
                with open(self.right_camera_yaml, 'r') as f:
                    cam_r = yaml.safe_load(f)
                P_r = cam_r.get('projection_matrix', {}).get('data', None)
                if P_r and len(P_r) == 12:
                    Tx_right = float(P_r[3])
                rospy.loginfo('Loaded right projection from %s: Tx_right=%s', self.right_camera_yaml, str(Tx_right))
            except Exception as e:
                rospy.logwarn('Failed to load right camera YAML %s: %s', self.right_camera_yaml, str(e))

        # Compute baseline preference: both cameras -> (Tx_left - Tx_right)/fx; else single-camera fallback
        if self.fx > 0:
            if Tx_left is not None and Tx_right is not None:
                self.baseline = (Tx_left - Tx_right) / self.fx
                rospy.loginfo('Derived baseline from both cameras: (Tx_left - Tx_right)/fx = (%.6f - %.6f)/%.6f = %.6f m', Tx_left, Tx_right, self.fx, self.baseline)
            elif Tx_right is not None:
                # Standard right P has Tx = -fx * baseline
                self.baseline = -Tx_right / self.fx
                rospy.loginfo('Derived baseline from right camera only: -Tx_right/fx = -%.6f/%.6f = %.6f m', Tx_right, self.fx, self.baseline)
            elif Tx_left is not None:
                # Some setups encode Tx in left P
                self.baseline = -Tx_left / self.fx
                rospy.loginfo('Derived baseline from left camera only: -Tx_left/fx = -%.6f/%.6f = %.6f m', Tx_left, self.fx, self.baseline)

        # Publishers
        self.flow_pub = rospy.Publisher('~flow', Image, queue_size=1)
        self.conf_pub = rospy.Publisher('~confidence', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('~depth', Image, queue_size=1)
        self.depth_raw_pub = rospy.Publisher('~depth_raw', Image, queue_size=1) if self.publish_raw_depth else None
        self.pc_pub = rospy.Publisher('~points', PointCloud2, queue_size=1)

        # Build RAFTStereo args-like object
        class Args:
            pass
        args = Args()
        args.mixed_precision = self.mixed_precision
        args.valid_iters = self.valid_iters
        args.hidden_dims = self.hidden_dims
        args.corr_implementation = self.corr_implementation
        args.shared_backbone = self.shared_backbone
        args.corr_levels = self.corr_levels
        args.corr_radius = self.corr_radius
        args.n_downsample = self.n_downsample
        args.context_norm = self.context_norm
        args.slow_fast_gru = self.slow_fast_gru
        args.n_gru_layers = self.n_gru_layers

        # Load model
        self.model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        try:
            state = torch.load(self.restore_ckpt, map_location=self.device)
            if 'module.' not in next(iter(state.keys())):
                state = {f'module.{k}': v for k, v in state.items()}
            self.model.load_state_dict(state, strict=self.strict_load)
        except Exception as e:
            rospy.logerr('Failed to load checkpoint %s with strict=%s: %s', self.restore_ckpt, str(self.strict_load), str(e))
            raise
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

        # Log calibration summary
        rospy.loginfo('Calibration: fx=%.3f fy=%.3f cx=%.3f cy=%.3f baseline=%.6f m', self.fx, self.fy, self.cx, self.cy, self.baseline)
        rospy.loginfo('Point cloud params: step=%d use_masked=%s max_depth=%.2f depth_scale=%.3f', self.pc_step, str(self.pc_use_masked_depth), self.max_depth, self.depth_scale)

        # Warm-up
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256).to(self.device)
            padder = InputPadder(dummy.shape, divis_by=32)
            d1, d2 = padder.pad(dummy, dummy)
            _ = self.model(d1, d2, iters=self.valid_iters, test_mode=True)

        # Subscribers with approximate time sync
        left_sub = message_filters.Subscriber(self.left_topic, Image)
        right_sub = message_filters.Subscriber(self.right_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.callback)

    def _rosimg_to_torch(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if cv_img.ndim == 2:
            cv_img = np.stack([cv_img, cv_img, cv_img], axis=-1)
        elif cv_img.shape[2] == 4:
            cv_img = cv_img[:, :, :3]
        if cv_img.dtype != np.uint8:
            cv_img = cv_img.astype(np.uint8)
        img = torch.from_numpy(cv_img).permute(2, 0, 1).float().unsqueeze(0)
        return img.to(self.device)

    def compute_depth_from_disp(self, disp_tensor, conf_tensor=None):
        # disp_tensor: [1,H,W] float tensor on any device
        # conf_tensor: [1,H,W] or None
        disp = disp_tensor.clone()
        # RAFTStereo returns negative disparity for right image shift; invert sign if needed
        # Use absolute disparity magnitude in denominator
        disp = disp.abs()
        disp = torch.clamp(disp, min=self.min_disp)

        if self.fx <= 0.0 or self.baseline <= 0.0:
            return None, None

        depth = (self.fx * self.baseline) / disp

        if conf_tensor is not None:
            mask = (conf_tensor >= self.conf_threshold).to(depth.dtype)
            depth_masked = depth * mask + (~(mask.bool())).to(depth.dtype) * np.float32(np.nan)
        else:
            depth_masked = depth

        return depth, depth_masked

    def depth_to_pointcloud(self, depth_np):
        # depth_np: HxW float32 in meters; NaNs invalid
        H, W = depth_np.shape
        step = max(1, int(self.pc_step))
        fy = self.fy if self.fy > 0 else self.fx
        cx = self.cx if self.cx != 0 else (W - 1) / 2.0
        cy = self.cy if self.cy != 0 else (H - 1) / 2.0

        us = np.arange(0, W, step, dtype=np.float32)
        vs = np.arange(0, H, step, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(us, vs)
        z = depth_np[::step, ::step]

        valid = np.isfinite(z) & (z > 0) & (z <= self.max_depth)
        if not np.any(valid):
            return None

        x = (u_grid - cx) * z / self.fx
        y = (v_grid - cy) * z / fy

        x = x[valid]
        y = y[valid]
        z = z[valid]

        points = np.vstack((x, y, z)).T.astype(np.float32)
        return points

    def callback(self, left_msg, right_msg):
        try:
            if self.debug:
                rospy.loginfo('Callback: left enc=%s right enc=%s', getattr(left_msg, 'encoding', 'n/a'), getattr(right_msg, 'encoding', 'n/a'))

            image1 = self._rosimg_to_torch(left_msg)
            image2 = self._rosimg_to_torch(right_msg)
            if self.debug:
                rospy.loginfo('Input shapes: %s %s', tuple(image1.shape), tuple(image2.shape))

            padder = InputPadder(image1.shape, divis_by=32)
            image1_p, image2_p = padder.pad(image1, image2)
            if self.debug:
                rospy.loginfo('Padded shapes: %s %s', tuple(image1_p.shape), tuple(image2_p.shape))

            with torch.no_grad():
                out = self.model(image1_p, image2_p, iters=self.valid_iters, test_mode=True)
                # Forward returns: flow_low, flow_up, corr_confidence, prior_confidence
                flow_low, flow_up, corr_confidence, _ = out

            flow_up = padder.unpad(flow_up).squeeze(0)
            if self.debug:
                rospy.loginfo('Flow shape after unpad: %s', tuple(flow_up.shape))

            if self.compute_confidence:
                # corr_confidence is [N,H,W]; add channel for unpad
                if corr_confidence.dim() == 3:
                    conf = padder.unpad(corr_confidence.unsqueeze(1)).squeeze(1)
                else:
                    conf = padder.unpad(corr_confidence)
                # conf is [N,H,W]; upsample if needed
                if conf.shape[-2:] != flow_up.shape[-2:]:
                    conf = torch.nn.functional.interpolate(conf.unsqueeze(1), size=flow_up.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)
                conf = conf.squeeze(0)
                conf_np = conf.detach().cpu().numpy().astype(np.float32)
            else:
                conf = None
                conf_np = None

            disp = flow_up[:, :, :]
            if disp.shape[0] > 1:
                disp = disp[0:1, :, :]
            disp_np = disp.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # Compute and publish depth
            depth, depth_masked = self.compute_depth_from_disp(disp.squeeze(0), conf)
            # Apply scaling
            if depth is not None:
                scale = float(self.depth_scale)
                depth = depth * scale
                if depth_masked is not None:
                    depth_masked = depth_masked * scale

            header = Header()
            header.stamp = left_msg.header.stamp
            header.frame_id = 'camera_depth_optical_frame'

            flow_msg = self.bridge.cv2_to_imgmsg(disp_np, encoding='32FC1')
            flow_msg.header = header
            self.flow_pub.publish(flow_msg)

            if conf_np is not None:
                conf_msg = self.bridge.cv2_to_imgmsg(conf_np, encoding='32FC1')
                conf_msg.header = header
                self.conf_pub.publish(conf_msg)

            if depth is None:
                if self.debug:
                    rospy.logwarn('Skipping depth/pointcloud: invalid fx(%.3f) or baseline(%.6f)', self.fx, self.baseline)
                return

            depth_np = depth.detach().cpu().numpy().astype(np.float32)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding='32FC1')
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)

            if self.depth_raw_pub is not None and depth_masked is not None:
                depth_masked_np = depth_masked.detach().cpu().numpy().astype(np.float32)
                depth_raw_msg = self.bridge.cv2_to_imgmsg(depth_masked_np, encoding='32FC1')
                depth_raw_msg.header = header
                self.depth_raw_pub.publish(depth_raw_msg)

            # Publish point cloud (use masked depth if configured)
            depth_for_pc = depth_masked.detach().cpu().numpy().astype(np.float32) if (self.pc_use_masked_depth and depth_masked is not None) else depth_np
            points = self.depth_to_pointcloud(depth_for_pc)
            if points is None or points.size == 0:
                if self.debug:
                    valid_count = np.isfinite(depth_for_pc).sum()
                    rospy.logwarn('No valid 3D points: valid depth count=%d (threshold=%.2f, use_masked=%s)', int(valid_count), self.conf_threshold, str(self.pc_use_masked_depth))
                return

            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
            pc_msg = pc2.create_cloud(header, fields, points)
            self.pc_pub.publish(pc_msg)
            if self.debug:
                rospy.loginfo('Published point cloud with %d points', points.shape[0])
        except Exception:
            rospy.logerr('RAFTStereo inference error:\n%s', traceback.format_exc())


def main():
    rospy.init_node('raftstereo_node')
    _ = RAFTStereoNode()
    rospy.loginfo('RAFTStereo node started')
    rospy.spin()


if __name__ == '__main__':
    main() 