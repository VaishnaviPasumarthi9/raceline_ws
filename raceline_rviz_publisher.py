#!/usr/bin/env python3
"""
raceline_rviz2_publisher.py
===========================
Publishes the computed raceline as RViz2-compatible ROS 2 topics.

Topics:
  /raceline/path        — nav_msgs/msg/Path
  /raceline/markers     — visualization_msgs/msg/MarkerArray  (line + speed dots)
  /raceline/arrows      — visualization_msgs/msg/MarkerArray  (velocity arrows)
  /raceline/boundaries  — visualization_msgs/msg/MarkerArray  (left/right walls)
  /raceline/labels      — visualization_msgs/msg/MarkerArray  (speed text)

Usage:
  source /opt/ros/jazzy/setup.bash
  python3 raceline_rviz2_publisher.py --csv output/raceline.csv [--frame map] [--loop] [--rate 0.5]
"""

import argparse
import math
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Vector3
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def speed_to_rgb(v: float, v_min: float, v_max: float):
    """Map speed scalar to red→yellow→green RGB."""
    t = (v - v_min) / (v_max - v_min + 1e-9)
    t = max(0.0, min(1.0, t))
    if t >= 0.5:
        r = 1.0 - 2.0 * (t - 0.5)
        g = 1.0
    else:
        r = 1.0
        g = 2.0 * t
    return r, g, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

class RacelinePublisher(Node):
    def __init__(self, csv_path, frame_id, loop, rate_hz):
        super().__init__("raceline_publisher")

        # Transient Local = latched (late-joining subscribers still receive)
        latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.pub_path    = self.create_publisher(Path,        "/raceline/path",       latching_qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/raceline/markers",    latching_qos)
        self.pub_arrows  = self.create_publisher(MarkerArray, "/raceline/arrows",     latching_qos)
        self.pub_bounds  = self.create_publisher(MarkerArray, "/raceline/boundaries", latching_qos)
        self.pub_labels  = self.create_publisher(MarkerArray, "/raceline/labels",     latching_qos)

        self.get_logger().info(f"Loading CSV: {csv_path}")
        self.frame_id = frame_id
        self.data     = self._load_csv(csv_path)

        # Normal vectors derived from heading angle (psi)
        # n = [-sin(psi), cos(psi)]  — points to the left of the heading
        psi = self.data["psi"]
        self.normvecs = np.column_stack([-np.sin(psi), np.cos(psi)])

        self.get_logger().info(
            f"Loaded {len(self.data['raceline'])} points | "
            f"speed {self.data['vx'].min():.2f}–{self.data['vx'].max():.2f} m/s"
        )

        if loop:
            self.timer = self.create_timer(1.0 / rate_hz, self._publish)
        else:
            self._publish()   # publish once; QoS latching keeps it alive

    # ── CSV loader ──────────────────────────────────────────────────────────

    def _load_csv(self, csv_path: str) -> dict:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        return {
            "raceline": data[:, 0:2],   # x_m, y_m
            "w_right":  data[:, 2],     # w_tr_right_m
            "w_left":   data[:, 3],     # w_tr_left_m
            "psi":      data[:, 4],     # psi_rad
            "kappa":    data[:, 5],     # kappa_radpm
            "vx":       data[:, 6],     # vx_mps
            "ax":       data[:, 7],     # ax_mps2
        }

    # ── Main publish callback ───────────────────────────────────────────────

    def _publish(self):
        stamp  = self.get_clock().now().to_msg()
        header = Header(frame_id=self.frame_id, stamp=stamp)

        self._publish_path(header)
        self._publish_speed_markers(header)
        self._publish_arrows(header)
        self._publish_boundaries(header)
        self._publish_labels(header)

    # ── 1. Path ─────────────────────────────────────────────────────────────

    def _publish_path(self, header):
        path_msg = Path(header=header)
        for i, (x, y) in enumerate(self.data["raceline"]):
            ps = PoseStamped(header=header)
            ps.pose.position    = Point(x=float(x), y=float(y), z=0.0)
            ps.pose.orientation = yaw_to_quaternion(self.data["psi"][i])
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)

    # ── 2. Speed markers (line strip + coloured spheres) ────────────────────

    def _publish_speed_markers(self, header):
        ma    = MarkerArray()
        v_min = float(self.data["vx"].min())
        v_max = float(self.data["vx"].max())

        # --- line strip (orange outline) ---
        line = Marker(
            header=header, ns="raceline_path", id=0,
            type=Marker.LINE_STRIP, action=Marker.ADD,
        )
        line.scale.x = 0.04
        line.color   = ColorRGBA(r=1.0, g=0.3, b=0.0, a=1.0)
        line.pose.orientation.w = 1.0
        for x, y in self.data["raceline"]:
            line.points.append(Point(x=float(x), y=float(y), z=0.02))
        line.points.append(line.points[0])   # close loop
        ma.markers.append(line)

        # --- speed-coloured spheres ---
        for i, (x, y) in enumerate(self.data["raceline"]):
            mk = Marker(
                header=header, ns="speed_dots", id=i + 1,
                type=Marker.SPHERE, action=Marker.ADD,
            )
            mk.pose.position = Point(x=float(x), y=float(y), z=0.05)
            mk.scale = Vector3(x=0.08, y=0.08, z=0.08)
            r, g, b  = speed_to_rgb(self.data["vx"][i], v_min, v_max)
            mk.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            ma.markers.append(mk)

        self.pub_markers.publish(ma)

    # ── 3. Velocity arrows ──────────────────────────────────────────────────

    def _publish_arrows(self, header):
        arrow_ma = MarkerArray()
        v_min    = float(self.data["vx"].min())
        v_max    = float(self.data["vx"].max())
        every_n  = 10

        for idx, i in enumerate(range(0, len(self.data["raceline"]), every_n)):
            x, y   = self.data["raceline"][i]
            length = float(0.5 * self.data["vx"][i] / (v_max + 1e-9))

            mk = Marker(
                header=header, ns="velocity_arrows", id=idx,
                type=Marker.ARROW, action=Marker.ADD,
            )
            mk.pose.position    = Point(x=float(x), y=float(y), z=0.1)
            mk.pose.orientation = yaw_to_quaternion(self.data["psi"][i])
            mk.scale = Vector3(x=max(length, 0.05), y=0.03, z=0.03)
            r, g, b  = speed_to_rgb(self.data["vx"][i], v_min, v_max)
            mk.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            arrow_ma.markers.append(mk)

        self.pub_arrows.publish(arrow_ma)

    # ── 4. Track boundaries ─────────────────────────────────────────────────

    def _publish_boundaries(self, header):
        bound_ma = MarkerArray()

        # right boundary: offset in -normal direction, left: +normal direction
        configs = [
            (self.data["w_right"], -1, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8), "boundary_right", 0),
            (self.data["w_left"],   1, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8), "boundary_left",  1),
        ]
        for w, sign, color, ns, s_id in configs:
            mk = Marker(
                header=header, ns=ns, id=s_id,
                type=Marker.LINE_STRIP, action=Marker.ADD,
            )
            mk.scale.x = 0.03
            mk.color   = color
            mk.pose.orientation.w = 1.0

            for i in range(len(self.data["raceline"])):
                bx = float(self.data["raceline"][i, 0] + sign * self.normvecs[i, 0] * w[i])
                by = float(self.data["raceline"][i, 1] + sign * self.normvecs[i, 1] * w[i])
                mk.points.append(Point(x=bx, y=by, z=0.01))
            mk.points.append(mk.points[0])   # close loop
            bound_ma.markers.append(mk)

        self.pub_bounds.publish(bound_ma)

    # ── 5. Speed text labels ─────────────────────────────────────────────────

    def _publish_labels(self, header):
        label_ma = MarkerArray()
        every_n  = 20

        for idx, i in enumerate(range(0, len(self.data["raceline"]), every_n)):
            mk = Marker(
                header=header, ns="speed_labels", id=idx,
                type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
            )
            mk.pose.position = Point(
                x=float(self.data["raceline"][i, 0]),
                y=float(self.data["raceline"][i, 1]),
                z=0.25,
            )
            mk.scale.z = 0.12
            mk.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            mk.text    = f"{self.data['vx'][i]:.1f} m/s"
            label_ma.markers.append(mk)

        self.pub_labels.publish(label_ma)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description="Publish raceline CSV → RViz2")
    parser.add_argument("--csv",   required=True,       help="Path to raceline.csv")
    parser.add_argument("--frame", default="map",        help="TF frame (default: map)")
    parser.add_argument("--loop",  action="store_true",  help="Re-publish periodically")
    parser.add_argument("--rate",  type=float, default=0.5, help="Rate in Hz (requires --loop)")

    # Strip ROS 2 remapping args before parsing
    parsed_args, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = RacelinePublisher(
        csv_path=parsed_args.csv,
        frame_id=parsed_args.frame,
        loop=parsed_args.loop,
        rate_hz=parsed_args.rate,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()