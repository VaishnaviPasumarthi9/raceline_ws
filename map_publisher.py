#!/usr/bin/env python3
"""
map_publisher.py  —  ROS2
==========================
Publishes a .pgm + .yaml map as nav_msgs/OccupancyGrid on /map.
No Nav2 lifecycle needed. Works directly with RViz2.

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 map_publisher.py \
        --pgm  maps/maps/test/my_map.pgm \
        --yaml maps/maps/test/my_map.yaml
"""

import argparse, sys, os
import numpy as np
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid, MapMetaData
    from std_msgs.msg import Header
    from geometry_msgs.msg import Pose, Point, Quaternion
    from builtin_interfaces.msg import Time
except ImportError:
    print("ERROR: rclpy not found.\n  source /opt/ros/jazzy/setup.bash")
    sys.exit(1)


def load_map(pgm_path, yaml_path):
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read: {pgm_path}")

    resolution = 0.05
    origin_x   = 0.0
    origin_y   = 0.0
    negate     = 0
    occ_thresh = 0.65
    free_thresh= 0.196

    if yaml_path and os.path.isfile(yaml_path):
        import yaml
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        resolution  = float(meta.get("resolution", resolution))
        origin      = meta.get("origin", [0, 0, 0])
        origin_x    = float(origin[0])
        origin_y    = float(origin[1])
        negate      = int(meta.get("negate", 0))
        occ_thresh  = float(meta.get("occupied_thresh", occ_thresh))
        free_thresh = float(meta.get("free_thresh",     free_thresh))

    return img, resolution, origin_x, origin_y, negate, occ_thresh, free_thresh


def pgm_to_occupancy(img, negate, occ_thresh, free_thresh):
    """
    Convert greyscale PGM to ROS OccupancyGrid data (0=free, 100=occupied, -1=unknown).
    ROS PGM convention (negate=0): white=free(254), black=occupied(0), grey=unknown(205)
    """
    h, w = img.shape
    data = np.full(h * w, -1, dtype=np.int8)   # unknown by default

    flat = img.flatten().astype(np.float32)

    if negate:
        flat = 255.0 - flat

    # Normalise to 0..1
    norm = flat / 255.0

    free_mask = norm >= (1.0 - free_thresh)      # bright = free
    occ_mask  = norm <= (1.0 - occ_thresh)       # dark   = occupied

    data[free_mask] = 0
    data[occ_mask]  = 100

    # ROS map origin is bottom-left but PGM row 0 is top — flip rows
    data_2d   = data.reshape(h, w)
    data_flip = np.flipud(data_2d)
    return data_flip.flatten().tolist()


class MapPublisher(Node):
    def __init__(self, pgm_path, yaml_path):
        super().__init__("map_publisher")

        # Latched QoS — RViz gets the map even if it subscribes late
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.pub = self.create_publisher(OccupancyGrid, "/map", latched)

        self.get_logger().info(f"Loading map: {pgm_path}")
        img, res, ox, oy, neg, occ_t, free_t = load_map(pgm_path, yaml_path)
        h, w = img.shape
        self.get_logger().info(f"Map: {w}x{h} px @ {res} m/px  origin=({ox:.3f},{oy:.3f})")

        # Build OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp    = self.get_clock().now().to_msg()

        msg.info.resolution = res
        msg.info.width      = w
        msg.info.height     = h
        msg.info.origin.position.x    = ox
        msg.info.origin.position.y    = oy
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = pgm_to_occupancy(img, neg, occ_t, free_t)

        self.msg = msg
        self.publish()

        # Re-publish every 2 s so late-joining RViz always sees it
        self.create_timer(2.0, self.publish)
        self.get_logger().info("Publishing /map  (latched, re-publishing every 2s)")

    def publish(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def main():
    parser = argparse.ArgumentParser(description="Publish PGM map as /map topic")
    parser.add_argument("--pgm",  required=True, help="Path to .pgm map image")
    parser.add_argument("--yaml", required=True, help="Path to .yaml map metadata")
    clean = [a for a in sys.argv[1:] if not a.startswith("__")]
    args  = parser.parse_args(clean)

    rclpy.init()
    node = MapPublisher(args.pgm, args.yaml)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
