#!/usr/bin/env python3
"""
pbstream_to_map.py
==================
Converts a Cartographer .pbstream file → my_map.pgm + my_map.yaml
WITHOUT needing cartographer_pbstream_to_ros_map (which crashes on
unfinished submaps / certain pbstream formats).

Works purely in Python. No ROS required for this step.

Usage:
    python3 pbstream_to_map.py \
        --pbstream maps/your_map.pbstream \
        --output   maps/my_map \
        --resolution 0.05
"""

import argparse
import math
import os
import struct
import sys
import zlib

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Protobuf varint / field parser  (no generated code needed)
# ─────────────────────────────────────────────────────────────────────────────

def _read_varint(data: bytes, pos: int):
    result = 0
    shift  = 0
    while pos < len(data):
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        shift  += 7
        if not (b & 0x80):
            break
    return result, pos


def _parse_proto_fields(data: bytes):
    """
    Minimal protobuf parser.
    Returns dict: field_number → list of raw values (varint / bytes).
    Does NOT recurse — caller decides which fields to descend into.
    """
    fields = {}
    pos = 0
    n   = len(data)
    while pos < n:
        if pos >= n:
            break
        tag, pos = _read_varint(data, pos)
        field_num  = tag >> 3
        wire_type  = tag & 0x7

        if wire_type == 0:          # varint
            val, pos = _read_varint(data, pos)
        elif wire_type == 1:        # 64-bit
            val = data[pos:pos+8]; pos += 8
        elif wire_type == 2:        # length-delimited
            length, pos = _read_varint(data, pos)
            val = data[pos:pos+length]; pos += length
        elif wire_type == 5:        # 32-bit
            val = data[pos:pos+4]; pos += 4
        else:
            # unknown wire type — skip rest (can't recover)
            break

        fields.setdefault(field_num, []).append(val)

    return fields


# ─────────────────────────────────────────────────────────────────────────────
# pbstream record reader
# The file format is:
#   [8-byte magic] repeated { [4-byte big-endian length] [compressed proto] }
# ─────────────────────────────────────────────────────────────────────────────

PBSTREAM_MAGIC = b'\xce\x9f\xaa\x18\xd8\xbf\x02\x13'   # may vary; we skip

def _read_pbstream_records(path: str):
    """
    Yield raw (decompressed) record bytes from a Cartographer pbstream.
    Handles both compressed and uncompressed records automatically.
    """
    with open(path, 'rb') as f:
        raw = f.read()

    pos = 0
    # Skip magic header if present
    if raw[:8] == PBSTREAM_MAGIC:
        pos = 8

    records = []
    while pos + 4 <= len(raw):
        # Try big-endian uint32 length prefix
        length = struct.unpack_from('>I', raw, pos)[0]
        pos += 4

        if length == 0 or pos + length > len(raw):
            # Try little-endian
            pos -= 4
            length = struct.unpack_from('<I', raw, pos)[0]
            pos += 4

        if length == 0 or pos + length > len(raw):
            break

        chunk = raw[pos:pos+length]
        pos  += length

        # Try decompressing with zlib
        try:
            chunk = zlib.decompress(chunk)
        except Exception:
            pass    # already uncompressed

        records.append(chunk)

    return records, raw   # also return raw for PNG search fallback


def _find_png_blobs(raw: bytes):
    """Find all PNG images embedded anywhere in the raw file bytes."""
    PNG_SIG = b'\x89PNG\r\n\x1a\n'
    images  = []
    idx     = 0
    while True:
        pos = raw.find(PNG_SIG, idx)
        if pos == -1:
            break
        # PNGs end with IEND chunk: b'IEND\xaeB`\x82'
        iend = raw.find(b'IEND\xaeB`\x82', pos)
        if iend == -1:
            end = min(pos + 20_000_000, len(raw))
        else:
            end = iend + 8
        blob = raw[pos:end]
        arr  = np.frombuffer(blob, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 500:
            images.append(img)
            print(f"  Found embedded PNG: {img.shape[1]}×{img.shape[0]} px at offset {pos}")
        idx = pos + 1
    return images


# ─────────────────────────────────────────────────────────────────────────────
# Cartographer submap grid extraction
#
# Cartographer proto layout (simplified):
#   SerializedData {
#     oneof {
#       Submap submap = 2;
#       ...
#     }
#   }
#   Submap {
#     Submap2D submap_2d = 2;
#   }
#   Submap2D {
#     Grid2D grid = 4;  (or probability_grid = older field 2)
#   }
#   Grid2D / ProbabilityGrid {
#     MapLimits limits = 1;
#     repeated float cells = 3 or 6;  (packed float or uint16)
#     ...
#   }
#   MapLimits {
#     double resolution = 1;
#     RigidTransform2D max = 2;  {x,y}
#     CellLimits cell_limits = 3; {num_x, num_y}
#   }
# ─────────────────────────────────────────────────────────────────────────────

def _decode_float_le(b4: bytes) -> float:
    return struct.unpack('<f', b4)[0]


def _try_extract_grid_from_record(rec: bytes):
    """
    Attempt to extract a probability grid image from one proto record.
    Returns (grid_img, resolution, origin_x, origin_y) or None.
    """
    try:
        top = _parse_proto_fields(rec)
    except Exception:
        return None

    # SerializedData.submap = field 2
    for submap_blob in top.get(2, []):
        if not isinstance(submap_blob, bytes) or len(submap_blob) < 4:
            continue
        try:
            submap_fields = _parse_proto_fields(submap_blob)
        except Exception:
            continue

        # Submap.submap_2d = field 2
        for s2d_blob in submap_fields.get(2, []):
            if not isinstance(s2d_blob, bytes):
                continue
            try:
                s2d = _parse_proto_fields(s2d_blob)
            except Exception:
                continue

            # Grid2D = field 4  OR ProbabilityGrid = field 2
            for grid_field in [4, 2]:
                for grid_blob in s2d.get(grid_field, []):
                    result = _decode_grid(grid_blob)
                    if result is not None:
                        return result
    return None


def _decode_grid(grid_blob: bytes):
    """
    Decode a Grid2D / ProbabilityGrid proto blob into a numpy image.
    Returns (img_uint8, resolution, origin_x, origin_y) or None.
    """
    if not isinstance(grid_blob, bytes) or len(grid_blob) < 8:
        return None
    try:
        g = _parse_proto_fields(grid_blob)
    except Exception:
        return None

    # limits = field 1
    limits_blobs = g.get(1, [])
    if not limits_blobs:
        return None

    resolution = 0.05
    num_x = num_y = 0
    max_x = max_y = 0.0

    for lim_blob in limits_blobs:
        if not isinstance(lim_blob, bytes):
            continue
        try:
            lim = _parse_proto_fields(lim_blob)
        except Exception:
            continue

        # resolution = field 1 (double)
        for rv in lim.get(1, []):
            if isinstance(rv, bytes) and len(rv) == 8:
                resolution = struct.unpack('<d', rv)[0]
            elif isinstance(rv, int):
                resolution = rv * 1e-9   # sometimes stored as fixed-point

        # max = field 2 (RigidTransform2D with x,y)
        for max_blob in lim.get(2, []):
            if isinstance(max_blob, bytes):
                try:
                    mb = _parse_proto_fields(max_blob)
                    for xv in mb.get(1, []):
                        if isinstance(xv, bytes) and len(xv) == 8:
                            max_x = struct.unpack('<d', xv)[0]
                    for yv in mb.get(2, []):
                        if isinstance(yv, bytes) and len(yv) == 8:
                            max_y = struct.unpack('<d', yv)[0]
                except Exception:
                    pass

        # cell_limits = field 3 {num_x=1, num_y=2}
        for cl_blob in lim.get(3, []):
            if isinstance(cl_blob, bytes):
                try:
                    cl = _parse_proto_fields(cl_blob)
                    for v in cl.get(1, []):
                        if isinstance(v, int): num_x = v
                    for v in cl.get(2, []):
                        if isinstance(v, int): num_y = v
                except Exception:
                    pass

    if num_x <= 0 or num_y <= 0:
        return None

    # cells: try field 3 (packed uint16) or field 6 (packed float)
    cells = None
    for fid in [3, 6, 5, 4]:
        for cell_blob in g.get(fid, []):
            if isinstance(cell_blob, bytes) and len(cell_blob) >= num_x * num_y * 2:
                n_u16 = len(cell_blob) // 2
                if n_u16 == num_x * num_y:
                    arr = np.frombuffer(cell_blob, dtype='<u2').astype(np.float32)
                    # Cartographer stores (probability - min) / (max - min) * 32767
                    cells = (arr / 32767.0)
                    break
                n_f32 = len(cell_blob) // 4
                if n_f32 == num_x * num_y:
                    cells = np.frombuffer(cell_blob, dtype='<f4').copy()
                    break
        if cells is not None:
            break

    if cells is None:
        return None

    cells = cells.reshape(num_y, num_x)
    # Convert probability to greyscale: free=254, occupied=0, unknown=205
    img = np.full((num_y, num_x), 205, dtype=np.uint8)
    img[cells > 0.65] = 0          # occupied
    img[cells < 0.35] = 254        # free
    img[(cells >= 0.35) & (cells <= 0.65)] = 205   # unknown

    origin_x = max_x - num_x * resolution
    origin_y = max_y - num_y * resolution

    return img, resolution, origin_x, origin_y


# ─────────────────────────────────────────────────────────────────────────────
# Merge multiple submap images into one combined map
# ─────────────────────────────────────────────────────────────────────────────

def merge_submaps(submaps):
    """
    Merge a list of (img, res, ox, oy) tuples into a single occupancy grid.
    """
    if len(submaps) == 1:
        return submaps[0]

    res = submaps[0][1]   # assume same resolution

    # World bounding box
    min_x = min(ox for _, _, ox, oy in submaps)
    min_y = min(oy for _, _, ox, oy in submaps)
    max_x = max(ox + img.shape[1]*res for img, _, ox, oy in submaps)
    max_y = max(oy + img.shape[0]*res for img, _, ox, oy in submaps)

    W = int(math.ceil((max_x - min_x) / res))
    H = int(math.ceil((max_y - min_y) / res))

    merged = np.full((H, W), 205, dtype=np.uint8)

    for img, _, ox, oy in submaps:
        px = int(round((ox - min_x) / res))
        py = int(round((max_y - (oy + img.shape[0]*res)) / res))
        h, w = img.shape

        # Clip to canvas
        sx, sy = 0, 0
        if px < 0: sx = -px; px = 0
        if py < 0: sy = -py; py = 0
        w = min(w - sx, W - px)
        h = min(h - sy, H - py)
        if w <= 0 or h <= 0:
            continue

        patch  = img [sy:sy+h, sx:sx+w]
        canvas = merged[py:py+h, px:px+w]

        # free overwrites unknown; occupied overwrites free and unknown
        mask_free = patch == 254
        mask_occ  = patch == 0
        canvas[mask_free & (canvas == 205)] = 254
        canvas[mask_occ]                    = 0
        merged[py:py+h, px:px+w]           = canvas

    return merged, res, min_x, min_y


# ─────────────────────────────────────────────────────────────────────────────
# YAML writer
# ─────────────────────────────────────────────────────────────────────────────

def write_yaml(yaml_path, pgm_filename, resolution, origin_x, origin_y):
    content = f"""image: {pgm_filename}
resolution: {resolution:.6f}
origin: [{origin_x:.6f}, {origin_y:.6f}, 0.000000]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"  YAML written: {yaml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def convert(pbstream_path: str, output_stem: str, resolution_override: float = None):
    pgm_path  = output_stem + ".pgm"
    yaml_path = output_stem + ".yaml"

    print(f"\n[pbstream_to_map] Reading: {pbstream_path}")
    records, raw = _read_pbstream_records(pbstream_path)
    print(f"  {len(records)} records found in pbstream")

    # ── Strategy 1: parse proto records ─────────────────────────────────────
    submaps = []
    for i, rec in enumerate(records):
        result = _try_extract_grid_from_record(rec)
        if result:
            img, res, ox, oy = result
            print(f"  Submap {len(submaps)+1}: {img.shape[1]}×{img.shape[0]} px  "
                  f"res={res:.4f}  origin=({ox:.2f},{oy:.2f})")
            submaps.append((img, resolution_override or res, ox, oy))

    if submaps:
        print(f"\n  Merging {len(submaps)} submap(s)...")
        merged, res, ox, oy = merge_submaps(submaps)
    else:
        # ── Strategy 2: find embedded PNGs ──────────────────────────────────
        print("  Proto parse found no grids — searching for embedded PNG blobs...")
        pngs = _find_png_blobs(raw)
        if not pngs:
            print("\nERROR: Could not extract any map data from this pbstream.")
            print("Please try exporting manually:")
            print("  ros2 run cartographer_ros cartographer_pbstream_to_ros_map \\")
            print(f"    -pbstream_filename {pbstream_path} \\")
            print(f"    -map_filestem {output_stem} -resolution 0.05")
            sys.exit(1)

        # Use largest PNG
        merged = max(pngs, key=lambda x: x.size)
        res    = resolution_override or 0.05
        ox, oy = 0.0, 0.0
        print(f"  Using largest embedded PNG: {merged.shape[1]}×{merged.shape[0]} px")

    if resolution_override:
        res = resolution_override

    # ── Write PGM ────────────────────────────────────────────────────────────
    # Flip vertically: ROS PGM convention has row 0 = bottom of world
    merged_flipped = cv2.flip(merged, 0)
    cv2.imwrite(pgm_path, merged_flipped)
    print(f"  PGM  written: {pgm_path}  ({merged.shape[1]}×{merged.shape[0]} px)")

    # ── Write YAML ───────────────────────────────────────────────────────────
    pgm_filename = os.path.basename(pgm_path)
    write_yaml(yaml_path, pgm_filename, res, ox, oy)

    # ── Preview ──────────────────────────────────────────────────────────────
    preview_path = output_stem + "_preview.png"
    preview = cv2.applyColorMap(merged_flipped, cv2.COLORMAP_BONE)
    cv2.imwrite(preview_path, preview)
    print(f"  Preview PNG: {preview_path}")

    print(f"\n[pbstream_to_map] Done.  Map size: {merged.shape[1]*res:.1f} × "
          f"{merged.shape[0]*res:.1f} m  @  {res} m/px")
    return pgm_path, yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cartographer pbstream → pgm + yaml (no ROS required)")
    parser.add_argument("--pbstream",   required=True, help="Input .pbstream file")
    parser.add_argument("--output",     required=True,
                        help="Output filestem (e.g. maps/my_map → my_map.pgm + my_map.yaml)")
    parser.add_argument("--resolution", type=float, default=None,
                        help="Override map resolution [m/px] (default: read from pbstream)")
    args = parser.parse_args()

    convert(args.pbstream, args.output, args.resolution)


if __name__ == "__main__":
    main()
