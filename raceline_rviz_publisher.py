import argparse
import math
import sys
import numpy as np

# ── ROS 2 imports ─────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from std_msgs.msg import Header, ColorRGBA
    from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
    from nav_msgs.msg import Path
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    print("ERROR: rclpy not found. Source your ROS 2 workspace first:\n"
          "  source /opt/ros/<distro>/setup.bash")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (Math remains the same)
# ──────────────────────────────────────────────────────────────────────────────

def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

def speed_to_rgb(v: float, v_min: float, v_max: float):
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
# Publisher Node
# ──────────────────────────────────────────────────────────────────────────────

class RacelinePublisher(Node):
    def __init__(self, csv_path, frame_id, loop, rate_hz):
        super().__init__('raceline_publisher')
        
        # QoS Profile to mimic "latch=True" (Transient Local)
        latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Publishers
        self.pub_path = self.create_publisher(Path, "/raceline/path", latch_qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/raceline/markers", latch_qos)
        self.pub_arrows = self.create_publisher(MarkerArray, "/raceline/arrows", latch_qos)
        self.pub_bounds = self.create_publisher(MarkerArray, "/raceline/boundaries", latch_qos)
        self.pub_labels = self.create_publisher(MarkerArray, "/raceline/labels", latch_qos)

        self.get_logger().info(f"Loading CSV: {csv_path}")
        self.data = self.load_raceline_csv(csv_path)
        self.normvecs = self.compute_normvecs_from_psi(self.data["psi"])
        self.frame_id = frame_id

        if loop:
            self.timer = self.create_timer(1.0 / rate_hz, self.publish_callback)
        else:
            # Publish once and keep node alive
            self.publish_callback()
            self.get_logger().info("Published once (Latched).")

    def load_raceline_csv(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        return {
            "raceline": data[:, 0:2],
            "w_right":  data[:, 2],
            "w_left":   data[:, 3],
            "psi":      data[:, 4],
            "vx":       data[:, 6],
        }

    def compute_normvecs_from_psi(self, psi: np.ndarray) -> np.ndarray:
        return np.column_stack([-np.sin(psi), np.cos(psi)])

    def publish_callback(self):
        stamp = self.get_clock().now().to_msg()
        d = self.data

        # 1. Path
        path_msg = Path()
        path_msg.header = Header(frame_id=self.frame_id, stamp=stamp)
        for i, (x, y) in enumerate(d["raceline"]):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position = Point(x=x, y=y, z=0.0)
            ps.pose.orientation = yaw_to_quaternion(d["psi"][i])
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)

        # 2. Markers (Dots + Line)
        ma = MarkerArray()
        v_min, v_max = d["vx"].min(), d["vx"].max()
        
        line_mk = Marker(header=Header(frame_id=self.frame_id, stamp=stamp), ns="path", id=0, 
                         type=Marker.LINE_STRIP, action=Marker.ADD)
        line_mk.scale.x = 0.05
        line_mk.color = ColorRGBA(r=1.0, g=0.3, b=0.0, a=1.0)
        for x, y in d["raceline"]:
            line_mk.points.append(Point(x=x, y=y, z=0.02))
        ma.markers.append(line_mk)

        for i, (x, y) in enumerate(d["raceline"]):
            if i % 2 != 0: continue # Optional: skip some dots for performance
            mk = Marker(header=Header(frame_id=self.frame_id, stamp=stamp), ns="dots", id=i+1,
                        type=Marker.SPHERE, action=Marker.ADD)
            mk.pose.position = Point(x=x, y=y, z=0.05)
            mk.scale = Vector3(x=0.1, y=0.1, z=0.1)
            r, g, b = speed_to_rgb(d["vx"][i], v_min, v_max)
            mk.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            ma.markers.append(mk)
        self.pub_markers.publish(ma)

        # 3. Boundaries
        bma = MarkerArray()
        for side_idx, (w, sign, color, ns) in enumerate([
            (d["w_right"], -1, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8), "right"),
            (d["w_left"],  +1, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8), "left"),
        ]):
            mk = Marker(header=Header(frame_id=self.frame_id, stamp=stamp), ns=ns, id=side_idx,
                        type=Marker.LINE_STRIP, action=Marker.ADD)
            mk.scale.x = 0.03
            mk.color = color
            for i in range(len(d["raceline"])):
                bx = d["raceline"][i, 0] + sign * self.normvecs[i, 0] * w[i]
                by = d["raceline"][i, 1] + sign * self.normvecs[i, 1] * w[i]
                mk.points.append(Point(x=bx, y=by, z=0.01))
            bma.markers.append(mk)
        self.pub_bounds.publish(bma)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--frame", default="map")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--rate", type=float, default=0.5)
    
    # Filter out ROS internal args
    ros_args = [a for a in sys.argv[1:] if not a.startswith("__")]
    args = parser.parse_args(ros_args)

    rclpy.init()
    node = RacelinePublisher(args.csv, args.frame, args.loop, args.rate)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()