#!/usr/bin/env python3
"""
pure_pursuit.py — ROS2 Pure Pursuit Controller for AutoDRIVE RoboRacer
=======================================================================
Reads raceline.csv and publishes AckermannDriveStamped to /drive.

THIS is what actually makes the car faster than FTG.
The raceline gives wider arcs through corners → higher corner speed.

Subscriptions:
  /ego_racecar/odom  OR  /odom   — nav_msgs/Odometry  (car position)

Publishes:
  /drive                          — ackermann_msgs/AckermannDriveStamped

Usage:
  source /opt/ros/jazzy/setup.bash
  pip3 install ackermann-msgs  # if not already installed
  python3 pure_pursuit.py --csv output/raceline.csv

Tuning for lap time:
  --lookahead   : smaller = tighter tracking, larger = smoother (default 0.8m)
  --speed_gain  : multiply all CSV speeds by this factor (default 1.0)
                  increase to go faster, decrease if car goes off track
"""

import argparse
import math
import sys
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
except ImportError:
    print("ERROR: rclpy not found.\n  source /opt/ros/jazzy/setup.bash")
    sys.exit(1)

try:
    from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
except ImportError:
    print("ERROR: ackermann_msgs not found.")
    print("  sudo apt install -y ros-jazzy-ackermann-msgs")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def quaternion_to_yaw(q):
    """Extract yaw from geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def find_lookahead_point(car_x, car_y, raceline_xy, lookahead_dist, last_idx):
    """
    Find the first raceline point that is at least lookahead_dist away
    from the car position, starting search from last_idx.
    Returns (target_x, target_y, new_idx).
    """
    n = len(raceline_xy)
    for i in range(n):
        idx = (last_idx + i) % n
        dx  = raceline_xy[idx, 0] - car_x
        dy  = raceline_xy[idx, 1] - car_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist >= lookahead_dist:
            return raceline_xy[idx, 0], raceline_xy[idx, 1], idx
    # fallback: closest point
    dists = np.sqrt((raceline_xy[:, 0] - car_x)**2 +
                    (raceline_xy[:, 1] - car_y)**2)
    idx = int(np.argmin(dists))
    return raceline_xy[idx, 0], raceline_xy[idx, 1], idx


def find_closest_idx(car_x, car_y, raceline_xy, last_idx, search_window=50):
    """Find the closest raceline point near last_idx (efficient)."""
    n   = len(raceline_xy)
    best_dist = float('inf')
    best_idx  = last_idx
    for i in range(search_window):
        idx  = (last_idx + i) % n
        dx   = raceline_xy[idx, 0] - car_x
        dy   = raceline_xy[idx, 1] - car_y
        dist = dx*dx + dy*dy
        if dist < best_dist:
            best_dist = dist
            best_idx  = idx
    return best_idx


def pure_pursuit_steering(car_x, car_y, car_yaw,
                           target_x, target_y,
                           lookahead_dist, wheelbase):
    """
    Pure Pursuit steering angle.
    Returns steering_angle in radians.
    """
    # Transform target to car frame
    dx = target_x - car_x
    dy = target_y - car_y
    # Rotate by -yaw
    local_x =  math.cos(car_yaw) * dx + math.sin(car_yaw) * dy
    local_y = -math.sin(car_yaw) * dx + math.cos(car_yaw) * dy

    # Pure pursuit formula: steering = atan(2 * L * sin(alpha) / ld)
    # where alpha = atan2(local_y, local_x)
    ld = math.sqrt(local_x**2 + local_y**2)
    if ld < 1e-6:
        return 0.0

    curvature    = 2.0 * local_y / (ld * ld)
    steering_rad = math.atan(curvature * wheelbase)
    return steering_rad


# ─────────────────────────────────────────────────────────────────────────────
# Pure Pursuit Node
# ─────────────────────────────────────────────────────────────────────────────

class PurePursuitNode(Node):

    def __init__(self, csv_path, lookahead, speed_gain, wheelbase,
                 max_steering_deg, odom_topic, drive_topic):
        super().__init__("pure_pursuit")

        # ── Load raceline ────────────────────────────────────────────────────
        self.get_logger().info(f"Loading raceline: {csv_path}")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.raceline_xy  = data[:, 0:2]          # x_m, y_m
        self.raceline_vx  = data[:, 6] * speed_gain  # vx_mps scaled
        self.raceline_vx  = np.clip(self.raceline_vx, 0.5, 20.0)
        self.n_pts        = len(self.raceline_xy)

        total_dist = np.sum(np.sqrt(
            np.diff(self.raceline_xy[:, 0])**2 +
            np.diff(self.raceline_xy[:, 1])**2))
        self.get_logger().info(
            f"Raceline: {self.n_pts} pts  "
            f"length={total_dist:.1f}m  "
            f"speed={self.raceline_vx.min():.1f}–{self.raceline_vx.max():.1f} m/s"
        )

        # ── Parameters ───────────────────────────────────────────────────────
        self.lookahead        = lookahead
        self.wheelbase        = wheelbase
        self.max_steer_rad    = math.radians(max_steering_deg)
        self.last_idx         = 0

        # ── Publishers / Subscribers ─────────────────────────────────────────
        best_effort_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)

        # Try odom topic — AutoDRIVE uses /ego_racecar/odom
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, best_effort_qos)

        self.get_logger().info(
            f"Listening on: {odom_topic}\n"
            f"Publishing to: {drive_topic}\n"
            f"Lookahead: {lookahead} m  |  Wheelbase: {wheelbase} m  |  "
            f"Speed gain: {speed_gain}x"
        )
        self.get_logger().info("Pure Pursuit running — car should follow raceline now.")

    # ── Odometry callback — runs at sensor rate ──────────────────────────────

    def odom_callback(self, msg: Odometry):
        # Car pose in world frame
        car_x   = msg.pose.pose.position.x
        car_y   = msg.pose.pose.position.y
        car_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        # ── Find closest point on raceline ───────────────────────────────────
        self.last_idx = find_closest_idx(
            car_x, car_y, self.raceline_xy, self.last_idx)

        # ── Speed at current position (from CSV) ─────────────────────────────
        target_speed = float(self.raceline_vx[self.last_idx])

        # ── Adaptive lookahead: scale with speed ─────────────────────────────
        # At high speed, look further ahead for stability
        adaptive_ld = max(self.lookahead,
                          self.lookahead * target_speed / 3.0)
        adaptive_ld = min(adaptive_ld, 3.0)   # cap at 3m

        # ── Find lookahead point ──────────────────────────────────────────────
        tgt_x, tgt_y, _ = find_lookahead_point(
            car_x, car_y, self.raceline_xy, adaptive_ld, self.last_idx)

        # ── Pure pursuit steering ─────────────────────────────────────────────
        steer = pure_pursuit_steering(
            car_x, car_y, car_yaw,
            tgt_x, tgt_y,
            adaptive_ld, self.wheelbase)

        # Clamp steering
        steer = max(-self.max_steer_rad, min(self.max_steer_rad, steer))

        # ── Slow down in sharp corners ────────────────────────────────────────
        # If steering is large, reduce speed proportionally
        steer_fraction = abs(steer) / (self.max_steer_rad + 1e-9)
        corner_factor  = 1.0 - 0.4 * steer_fraction   # up to 40% speed reduction
        cmd_speed      = target_speed * corner_factor

        # ── Publish AckermannDriveStamped ─────────────────────────────────────
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp    = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.speed          = float(cmd_speed)
        drive_msg.drive.steering_angle = float(steer)
        self.pub_drive.publish(drive_msg)

        # Debug every 50 callbacks (~1s at 50Hz)
        if not hasattr(self, '_dbg_count'):
            self._dbg_count = 0
        self._dbg_count += 1
        if self._dbg_count % 50 == 0:
            self.get_logger().info(
                f"pos=({car_x:.2f},{car_y:.2f})  "
                f"yaw={math.degrees(car_yaw):.1f}°  "
                f"steer={math.degrees(steer):.1f}°  "
                f"speed={cmd_speed:.2f} m/s  "
                f"wp={self.last_idx}/{self.n_pts}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pure Pursuit controller for AutoDRIVE RoboRacer")
    parser.add_argument("--csv",          required=True,
                        help="Path to raceline.csv")
    parser.add_argument("--lookahead",    type=float, default=0.8,
                        help="Base lookahead distance [m] (default: 0.8)")
    parser.add_argument("--speed_gain",   type=float, default=1.0,
                        help="Multiply raceline speeds by this factor (default: 1.0)")
    parser.add_argument("--wheelbase",    type=float, default=0.3302,
                        help="Car wheelbase [m] (RoboRacer default: 0.3302)")
    parser.add_argument("--max_steer",    type=float, default=24.0,
                        help="Max steering angle [deg] (default: 24.0)")
    parser.add_argument("--odom_topic",   default="/ego_racecar/odom",
                        help="Odometry topic (default: /ego_racecar/odom)")
    parser.add_argument("--drive_topic",  default="/drive",
                        help="Drive command topic (default: /drive)")

    parsed, _ = parser.parse_known_args()

    rclpy.init()
    node = PurePursuitNode(
        csv_path       = parsed.csv,
        lookahead      = parsed.lookahead,
        speed_gain     = parsed.speed_gain,
        wheelbase      = parsed.wheelbase,
        max_steering_deg = parsed.max_steer,
        odom_topic     = parsed.odom_topic,
        drive_topic    = parsed.drive_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before exiting
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        node.pub_drive.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
