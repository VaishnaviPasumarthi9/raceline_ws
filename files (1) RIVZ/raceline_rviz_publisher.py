#!/usr/bin/env python3
"""
raceline_rviz_publisher.py
===========================
Publishes the computed raceline as RViz-compatible ROS topics.

Topics published:
  /raceline/path          — nav_msgs/Path          (the geometric line)
  /raceline/markers       — visualization_msgs/MarkerArray  (coloured speed spheres)
  /raceline/centerline    — nav_msgs/Path          (reference centreline)
  /raceline/boundaries    — visualization_msgs/MarkerArray  (left/right walls)

Usage:
  # Terminal 1 — make sure roscore is running
  roscore

  # Terminal 2 — run the publisher
  python raceline_rviz_publisher.py \
      --csv output/raceline.csv \
      --frame map \
      --loop          # keeps republishing every 2s so RViz sees it on connect

  # Then open RViz and add:
  #   - By topic -> /raceline/path         (Path display)
  #   - By topic -> /raceline/markers      (MarkerArray display)
  #   - By topic -> /raceline/boundaries   (MarkerArray display)
"""

import argparse
import math
import sys
import numpy as np

# ── ROS imports ───────────────────────────────────────────────────────────────
try:
    import rospy
    from std_msgs.msg import Header, ColorRGBA
    from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
    from nav_msgs.msg import Path
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    print("ERROR: rospy not found. Source your ROS workspace first:\n"
          "  source /opt/ros/<distro>/setup.bash")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert a yaw angle [rad] to a geometry_msgs/Quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def speed_to_rgb(v: float, v_min: float, v_max: float):
    """
    Map a speed value to an RGB colour using a green→yellow→red gradient.
    Fast = green (0,1,0), Medium = yellow (1,1,0), Slow = red (1,0,0).
    """
    t = (v - v_min) / (v_max - v_min + 1e-9)   # 0..1
    t = max(0.0, min(1.0, t))
    if t >= 0.5:
        r = 1.0 - 2.0 * (t - 0.5)
        g = 1.0
    else:
        r = 1.0
        g = 2.0 * t
    return r, g, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Message builders
# ──────────────────────────────────────────────────────────────────────────────

def build_path_msg(xy: np.ndarray,
                   psi: np.ndarray,
                   frame_id: str,
                   stamp) -> Path:
    """Build a nav_msgs/Path from Nx2 xy and N-dim psi arrays."""
    path = Path()
    path.header = Header(frame_id=frame_id, stamp=stamp)
    for i, (x, y) in enumerate(xy):
        ps = PoseStamped()
        ps.header = path.header
        ps.pose.position    = Point(x=x, y=y, z=0.0)
        ps.pose.orientation = yaw_to_quaternion(psi[i])
        path.poses.append(ps)
    return path


def build_speed_markers(raceline: np.ndarray,
                        vx:       np.ndarray,
                        frame_id: str,
                        stamp,
                        sphere_scale: float = 0.08) -> MarkerArray:
    """
    A MarkerArray of small spheres colour-coded by speed.
    Green = fast, Yellow = medium, Red = slow.
    """
    ma    = MarkerArray()
    v_min = vx.min()
    v_max = vx.max()

    # One LINE_STRIP marker for the path itself (efficient)
    line_mk = Marker()
    line_mk.header      = Header(frame_id=frame_id, stamp=stamp)
    line_mk.ns          = "raceline_path"
    line_mk.id          = 0
    line_mk.type        = Marker.LINE_STRIP
    line_mk.action      = Marker.ADD
    line_mk.scale.x     = 0.04           # line width [m]
    line_mk.color       = ColorRGBA(r=1.0, g=0.3, b=0.0, a=1.0)  # orange
    line_mk.pose.orientation.w = 1.0
    for x, y in raceline:
        line_mk.points.append(Point(x=x, y=y, z=0.02))
    # close loop
    line_mk.points.append(Point(x=raceline[0, 0], y=raceline[0, 1], z=0.02))
    ma.markers.append(line_mk)

    # Speed spheres
    for i, (x, y) in enumerate(raceline):
        mk = Marker()
        mk.header      = Header(frame_id=frame_id, stamp=stamp)
        mk.ns          = "speed_dots"
        mk.id          = i + 1
        mk.type        = Marker.SPHERE
        mk.action      = Marker.ADD
        mk.pose.position    = Point(x=x, y=y, z=0.05)
        mk.pose.orientation = Quaternion(w=1.0)
        mk.scale        = Vector3(x=sphere_scale, y=sphere_scale, z=sphere_scale)
        r, g, b         = speed_to_rgb(vx[i], v_min, v_max)
        mk.color        = ColorRGBA(r=r, g=g, b=b, a=0.9)
        ma.markers.append(mk)

    return ma


def build_velocity_arrow_markers(raceline: np.ndarray,
                                  psi:     np.ndarray,
                                  vx:      np.ndarray,
                                  frame_id: str,
                                  stamp,
                                  every_n: int = 10) -> MarkerArray:
    """
    Arrows every `every_n` points, length proportional to speed.
    Useful for seeing speed distribution at a glance.
    """
    ma    = MarkerArray()
    v_max = vx.max()

    for idx, i in enumerate(range(0, len(raceline), every_n)):
        x, y = raceline[i]
        yaw  = psi[i]
        spd  = vx[i]
        length = 0.5 * spd / v_max    # normalised arrow length [m]

        mk = Marker()
        mk.header      = Header(frame_id=frame_id, stamp=stamp)
        mk.ns          = "velocity_arrows"
        mk.id          = idx
        mk.type        = Marker.ARROW
        mk.action      = Marker.ADD
        mk.pose.position    = Point(x=x, y=y, z=0.1)
        mk.pose.orientation = yaw_to_quaternion(yaw)
        mk.scale        = Vector3(x=length, y=0.03, z=0.03)
        r, g, b         = speed_to_rgb(spd, vx.min(), vx.max())
        mk.color        = ColorRGBA(r=r, g=g, b=b, a=1.0)
        ma.markers.append(mk)

    return ma


def build_boundary_markers(raceline:  np.ndarray,
                            normvecs:  np.ndarray,
                            w_right:   np.ndarray,
                            w_left:    np.ndarray,
                            frame_id:  str,
                            stamp) -> MarkerArray:
    """
    Two LINE_STRIP markers showing the left and right track boundaries
    projected from the raceline + normal vectors.
    """
    ma = MarkerArray()

    for side_idx, (w, sign, color, ns) in enumerate([
        (w_right, -1, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8), "boundary_right"),
        (w_left,  +1, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8), "boundary_left"),
    ]):
        mk = Marker()
        mk.header      = Header(frame_id=frame_id, stamp=stamp)
        mk.ns          = ns
        mk.id          = side_idx
        mk.type        = Marker.LINE_STRIP
        mk.action      = Marker.ADD
        mk.scale.x     = 0.03
        mk.color       = color
        mk.pose.orientation.w = 1.0

        for i in range(len(raceline)):
            bx = raceline[i, 0] + sign * normvecs[i, 0] * w[i]
            by = raceline[i, 1] + sign * normvecs[i, 1] * w[i]
            mk.points.append(Point(x=bx, y=by, z=0.01))
        # close loop
        bx = raceline[0, 0] + sign * normvecs[0, 0] * w[0]
        by = raceline[0, 1] + sign * normvecs[0, 1] * w[0]
        mk.points.append(Point(x=bx, y=by, z=0.01))

        ma.markers.append(mk)

    return ma


def build_text_markers(raceline: np.ndarray,
                       vx:       np.ndarray,
                       frame_id: str,
                       stamp,
                       every_n:  int = 20) -> MarkerArray:
    """Speed labels at regular intervals along the raceline."""
    ma = MarkerArray()
    for idx, i in enumerate(range(0, len(raceline), every_n)):
        mk = Marker()
        mk.header      = Header(frame_id=frame_id, stamp=stamp)
        mk.ns          = "speed_labels"
        mk.id          = idx
        mk.type        = Marker.TEXT_VIEW_FACING
        mk.action      = Marker.ADD
        mk.pose.position    = Point(x=raceline[i, 0],
                                    y=raceline[i, 1],
                                    z=0.25)
        mk.pose.orientation = Quaternion(w=1.0)
        mk.scale.z          = 0.12
        mk.color            = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        mk.text             = f"{vx[i]:.1f} m/s"
        ma.markers.append(mk)
    return ma


# ──────────────────────────────────────────────────────────────────────────────
# Main publisher
# ──────────────────────────────────────────────────────────────────────────────

def load_raceline_csv(csv_path: str):
    """
    Load the raceline CSV written by raceline_generator.py.
    Columns: x_m, y_m, w_tr_right_m, w_tr_left_m, psi_rad,
             kappa_radpm, vx_mps, ax_mps2
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return {
        "raceline": data[:, 0:2],
        "w_right":  data[:, 2],
        "w_left":   data[:, 3],
        "psi":      data[:, 4],
        "kappa":    data[:, 5],
        "vx":       data[:, 6],
        "ax":       data[:, 7],
    }


def compute_normvecs_from_psi(psi: np.ndarray) -> np.ndarray:
    """Derive left-normal vectors from heading angles."""
    return np.column_stack([-np.sin(psi), np.cos(psi)])


def publish_raceline(csv_path:   str,
                     frame_id:   str  = "map",
                     loop:       bool = True,
                     rate_hz:    float = 0.5):

    rospy.init_node("raceline_publisher", anonymous=False)

    pub_path     = rospy.Publisher("/raceline/path",       Path,        queue_size=1, latch=True)
    pub_markers  = rospy.Publisher("/raceline/markers",    MarkerArray, queue_size=1, latch=True)
    pub_arrows   = rospy.Publisher("/raceline/arrows",     MarkerArray, queue_size=1, latch=True)
    pub_bounds   = rospy.Publisher("/raceline/boundaries", MarkerArray, queue_size=1, latch=True)
    pub_labels   = rospy.Publisher("/raceline/labels",     MarkerArray, queue_size=1, latch=True)

    rospy.loginfo(f"[raceline_publisher] Loading: {csv_path}")
    d        = load_raceline_csv(csv_path)
    normvecs = compute_normvecs_from_psi(d["psi"])

    rate = rospy.Rate(rate_hz)
    rospy.loginfo("[raceline_publisher] Publishing raceline. "
                  "Add topics in RViz:\n"
                  "  /raceline/path        → Path\n"
                  "  /raceline/markers     → MarkerArray  (speed dots + line)\n"
                  "  /raceline/arrows      → MarkerArray  (velocity arrows)\n"
                  "  /raceline/boundaries  → MarkerArray  (track edges)\n"
                  "  /raceline/labels      → MarkerArray  (speed text)")

    while not rospy.is_shutdown():
        stamp = rospy.Time.now()

        path_msg    = build_path_msg(d["raceline"], d["psi"], frame_id, stamp)
        speed_mk    = build_speed_markers(d["raceline"], d["vx"], frame_id, stamp)
        arrow_mk    = build_velocity_arrow_markers(d["raceline"], d["psi"],
                                                   d["vx"], frame_id, stamp)
        bound_mk    = build_boundary_markers(d["raceline"], normvecs,
                                              d["w_right"], d["w_left"],
                                              frame_id, stamp)
        label_mk    = build_text_markers(d["raceline"], d["vx"], frame_id, stamp)

        pub_path.publish(path_msg)
        pub_markers.publish(speed_mk)
        pub_arrows.publish(arrow_mk)
        pub_bounds.publish(bound_mk)
        pub_labels.publish(label_mk)

        if not loop:
            rospy.loginfo("[raceline_publisher] Published once. Keeping node alive (latched).")
            rospy.spin()
            break

        rate.sleep()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Publish raceline CSV → RViz topics")
    parser.add_argument("--csv",      required=True,
                        help="Path to raceline.csv (output of raceline_generator.py)")
    parser.add_argument("--frame",    default="map",
                        help="TF frame to publish in (default: map)")
    parser.add_argument("--loop",     action="store_true",
                        help="Re-publish at 0.5 Hz (useful when RViz starts late)")
    parser.add_argument("--rate",     type=float, default=0.5,
                        help="Re-publish rate [Hz] when --loop is set")
    # Remove ROS remapping args that argparse doesn't know about
    import sys
    args = parser.parse_args([a for a in sys.argv[1:] if not a.startswith("__")])

    publish_raceline(
        csv_path = args.csv,
        frame_id = args.frame,
        loop     = args.loop,
        rate_hz  = args.rate,
    )


if __name__ == "__main__":
    main()
