"""
Minimum Curvature Raceline Generator
=====================================
Input  : Cartographer .pbstream file (or pre-exported occupancy grid .pgm/.png + .yaml)
Output : raceline.csv  — columns: x_m, y_m, w_tr_right_m, w_tr_left_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
         raceline.png  — visualisation

Algorithm follows Heilmeier et al. (2020) / ForzaETH Race Stack global planner:
  1. Parse map from pbstream (via cartographer_pbstream_to_ros_map or direct protobuf)
  2. Extract occupancy grid → binary drivable mask
  3. Morphological skeleton → centreline
  4. Compute track half-widths along normal vectors
  5. Iterative Quadratic Programming minimum-curvature optimisation
  6. Forward–backward velocity profile solver
  7. Write outputs
"""

import os, sys, argparse, math, warnings
import numpy as np
import cv2
from scipy import ndimage, interpolate
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# QP solver – try quadprog first, fall back to osqp
try:
    import quadprog
    _QP_BACKEND = "quadprog"
except ImportError:
    try:
        import osqp, scipy.sparse as sp
        _QP_BACKEND = "osqp"
    except ImportError:
        _QP_BACKEND = None

# ──────────────────────────────────────────────────────────────────────────────
# 0.  PBSTREAM PARSING
# ──────────────────────────────────────────────────────────────────────────────

def load_map_from_pbstream(pbstream_path: str):
    """
    Convert a Cartographer .pbstream to a numpy occupancy array.

    Strategy (in order of availability):
      A) cartographer_pbstream_to_ros_map CLI → reads the .pgm it writes
      B) Direct protobuf parse (no ROS required)
      C) Fallback: look for a .pgm / .png / .yaml sibling with the same stem
    """
    stem = os.path.splitext(pbstream_path)[0]

    # -- Strategy C first (cheap): sibling image already exported
    for ext in [".pgm", ".png"]:
        candidate = stem + ext
        if os.path.isfile(candidate):
            yaml_file = stem + ".yaml"
            print(f"[load_map] Found pre-exported image: {candidate}")
            return _load_pgm_yaml(candidate, yaml_file if os.path.isfile(yaml_file) else None)

    # -- Strategy A: use cartographer CLI if available
    try:
        import subprocess, shutil
        if shutil.which("cartographer_pbstream_to_ros_map"):
            pgm_out = stem + "_exported.pgm"
            yaml_out = stem + "_exported.yaml"
            result = subprocess.run(
                ["cartographer_pbstream_to_ros_map",
                 "-pbstream_filename", pbstream_path,
                 "-map_filestem", stem + "_exported"],
                capture_output=True, timeout=60
            )
            if result.returncode == 0 and os.path.isfile(pgm_out):
                print("[load_map] Converted via cartographer_pbstream_to_ros_map")
                return _load_pgm_yaml(pgm_out, yaml_out if os.path.isfile(yaml_out) else None)
    except Exception as e:
        print(f"[load_map] cartographer CLI failed: {e}")

    # -- Strategy B: raw protobuf
    try:
        return _load_map_from_pbstream_proto(pbstream_path)
    except Exception as e:
        print(f"[load_map] Protobuf parse failed: {e}")

    raise RuntimeError(
        "Could not load map from pbstream.\n"
        "Options:\n"
        "  1) Export manually:  cartographer_pbstream_to_ros_map "
        f"-pbstream_filename {pbstream_path} -map_filestem {stem}\n"
        "  2) Place a .pgm/.png (and optionally .yaml) with the same stem next to the pbstream.\n"
        "  3) Use --pgm flag directly: python raceline_generator.py --pgm map.pgm --yaml map.yaml"
    )


def _load_pgm_yaml(pgm_path: str, yaml_path=None):
    """Load a ROS-style .pgm occupancy image + optional .yaml metadata."""
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {pgm_path}")

    resolution = 0.05          # metres per pixel (default)
    origin_x   = 0.0
    origin_y   = 0.0
    occupied_thresh = 65       # pixel value above which cell is occupied
    free_thresh     = 196      # pixel value below which cell is free

    if yaml_path and os.path.isfile(yaml_path):
        import yaml
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        resolution      = float(meta.get("resolution", resolution))
        origin          = meta.get("origin", [0, 0, 0])
        origin_x, origin_y = float(origin[0]), float(origin[1])
        occupied_thresh = int(meta.get("occupied_thresh", 0.65) * 255)
        free_thresh     = int(meta.get("free_thresh",     0.196) * 255)

    # ROS convention: 0=occupied, 205=unknown, 254=free  (negate for cv display)
    # For .pgm exported by cartographer: white=free, black=occupied
    # We create a binary mask: True = drivable (free)
    drivable = img > free_thresh           # free cells
    occupied = img < occupied_thresh       # occupied (walls)
    # unknown stays False in drivable

    return {
        "grid":       img,
        "drivable":   drivable.astype(np.uint8) * 255,
        "resolution": resolution,
        "origin_x":   origin_x,
        "origin_y":   origin_y,
    }


def _load_map_from_pbstream_proto(pbstream_path: str):
    """
    Minimal protobuf parse to extract the 2D probability grid.
    Works without ROS; requires only the 'protobuf' Python package.
    The pbstream stores a SerializedData array; we find the submap grid entries.
    """
    try:
        from google.protobuf import descriptor_pb2   # noqa – just test import
    except ImportError:
        raise ImportError("protobuf package not installed: pip install protobuf")

    # Cartographer pbstream is a ProtoStreamContainerHeader + repeated records.
    # Each record: uint64 length (big-endian) + serialized FormatVersion/SerializedData proto.
    # We look for submap_2d entries which contain the probability grid image.

    import struct

    FREE_PROBABILITY    = 0.9    # Cartographer's free cell probability
    OCCUPIED_PROBABILITY= 0.1

    with open(pbstream_path, "rb") as f:
        raw = f.read()

    # Search for PNG magic bytes embedded in the protobuf blob (submaps are stored as PNG)
    PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
    images = []
    idx = 0
    while True:
        pos = raw.find(PNG_MAGIC, idx)
        if pos == -1:
            break
        # Try to decode from this offset
        try:
            nparr  = np.frombuffer(raw[pos:pos+10_000_000], dtype=np.uint8)
            img    = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 100:
                images.append(img)
        except Exception:
            pass
        idx = pos + 1

    if not images:
        raise RuntimeError("No embedded PNG submap found in pbstream")

    # Use the largest submap (most complete map)
    grid = max(images, key=lambda x: x.size)
    print(f"[proto] Extracted submap grid: {grid.shape}")
    drivable = (grid > 127).astype(np.uint8) * 255

    return {
        "grid":       grid,
        "drivable":   drivable,
        "resolution": 0.05,      # Cartographer default; adjust via --resolution
        "origin_x":   0.0,
        "origin_y":   0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MAP PRE-PROCESSING  →  TRACK BOUNDARY EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_map(map_data: dict, dilation_px: int = 3):
    """
    Clean the binary drivable mask:
      - morphological closing to fill gaps
      - remove small blobs
      - dilate walls slightly for safety margin
    Returns cleaned binary mask (uint8, 255=free).
    """
    mask = map_data["drivable"].copy()

    # Close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component (the track)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == largest).astype(np.uint8)) * 255

    # Erode slightly to push centreline away from walls
    if dilation_px > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (dilation_px*2+1, dilation_px*2+1))
        mask = cv2.erode(mask, kernel2, iterations=1)

    return mask


def extract_centreline(mask: np.ndarray, resolution: float):
    """
    Zhang–Suen skeletonisation → ordered centreline points in metres.
    Returns Nx2 array of (x, y) in world coordinates (pixel frame origin at top-left).
    """
    from skimage.morphology import skeletonize
    binary = (mask > 127).astype(np.uint8)
    skel   = skeletonize(binary).astype(np.uint8) * 255

    # Find non-zero pixels
    ys, xs = np.where(skel > 0)
    if len(xs) < 10:
        raise RuntimeError("Skeletonisation produced too few points – check map quality")

    pts = np.column_stack([xs, ys]).astype(float)

    # Order points along the track using nearest-neighbour walk
    pts_ordered = _order_points_nn(pts)

    # Convert pixel → metres  (flip y because image y-axis is downward)
    pts_m = pts_ordered.copy()
    pts_m[:, 0] =  pts_ordered[:, 0] * resolution
    pts_m[:, 1] = -pts_ordered[:, 1] * resolution   # flip y

    return pts_m, skel


def _order_points_nn(pts: np.ndarray) -> np.ndarray:
    """Order an unordered set of 2-D points along a closed loop via NN walk."""
    n   = len(pts)
    tree   = KDTree(pts)
    visited= np.zeros(n, dtype=bool)
    order  = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = order[-1]
        dists, idxs = tree.query(pts[current], k=min(20, n))
        for idx in idxs[1:]:
            if not visited[idx]:
                order.append(idx)
                visited[idx] = True
                break

    return pts[order]


def compute_track_widths(centreline_px: np.ndarray,
                         mask: np.ndarray,
                         resolution: float,
                         safety_margin_m: float = 0.1):
    """
    For each centreline point, cast rays left/right along the normal and
    find the distance to the nearest wall.  Returns (w_right, w_left) in metres.
    """
    # Recompute pixel coords from centreline (already in pixels from order step)
    # centreline_px comes from _order_points_nn which is in pixel space
    n  = len(centreline_px)
    w_right = np.zeros(n)
    w_left  = np.zeros(n)

    h, w = mask.shape

    for i in range(n):
        # Tangent vector (finite difference, cyclic)
        p_prev = centreline_px[(i - 1) % n]
        p_next = centreline_px[(i + 1) % n]
        tang   = p_next - p_prev
        tlen   = np.linalg.norm(tang)
        if tlen < 1e-9:
            w_right[i] = safety_margin_m
            w_left[i]  = safety_margin_m
            continue
        tang  /= tlen
        # Normal (perpendicular): right = (ty, -tx), left = (-ty, tx)  [image coords]
        normal_right = np.array([ tang[1], -tang[0]])
        normal_left  = np.array([-tang[1],  tang[0]])

        cx, cy = centreline_px[i]

        for sign, normal, store in [(1, normal_right, w_right),
                                    (1, normal_left,  w_left)]:
            dist = 0.0
            for step in range(1, 500):
                nx = int(round(cx + normal[0] * step))
                ny = int(round(cy + normal[1] * step))
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    dist = step * resolution
                    break
                if mask[ny, nx] < 127:      # hit a wall
                    dist = step * resolution
                    break
            else:
                dist = 500 * resolution
            store[i] = max(dist - safety_margin_m, 0.05)

    return w_right, w_left


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CENTRELINE SMOOTHING & RESAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def smooth_and_resample(pts_m: np.ndarray,
                        step_m: float = 0.1,
                        smooth_s: float = None):
    """
    Fit a periodic (closed) parametric spline to the centreline,
    then resample at uniform arc-length intervals of step_m.
    Returns (N×2 resampled points, spline tck object).
    """
    # Close the loop
    pts_closed = np.vstack([pts_m, pts_m[0]])
    x, y = pts_closed[:, 0], pts_closed[:, 1]

    # Chord-length parameterisation
    dx   = np.diff(x)
    dy   = np.diff(y)
    seg  = np.sqrt(dx**2 + dy**2)
    t    = np.concatenate([[0], np.cumsum(seg)])
    total_len = t[-1]

    # Smoothing parameter
    if smooth_s is None:
        smooth_s = total_len * 0.01

    tck, _ = interpolate.splprep([x, y], u=t, s=smooth_s, per=True, k=5)

    # Resample
    n_pts   = max(50, int(total_len / step_m))
    u_new   = np.linspace(0, total_len, n_pts, endpoint=False)
    xs, ys  = interpolate.splev(u_new, tck)

    return np.column_stack([xs, ys]), tck, total_len


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MINIMUM CURVATURE OPTIMISATION  (Heilmeier 2020 / ForzaETH)
# ──────────────────────────────────────────────────────────────────────────────

def _build_qp_matrices(refline: np.ndarray,
                       normvecs: np.ndarray,
                       w_right: np.ndarray,
                       w_left: np.ndarray,
                       kappa_bound: float = None):
    """
    Build the QP matrices for minimum-curvature optimisation.

    Optimise alpha ∈ R^N  (lateral offsets from reference line) such that
    the total squared curvature is minimised subject to track bounds.

    Race line point:  p_i = refline_i + alpha_i * normvec_i

    The curvature approximation used:
        kappa_i ≈  (p_{i-1} - 2*p_i + p_{i+1}) / step²   (2nd finite difference)
    Minimise  sum_i kappa_i²  →  alpha^T H alpha + f^T alpha

    Constraints:
        -w_right_i ≤ alpha_i ≤ w_left_i   (within track bounds)
    """
    n   = len(refline)
    dx  = np.diff(np.append(refline[:, 0], refline[0, 0]))
    dy  = np.diff(np.append(refline[:, 1], refline[1, 1]))
    ds  = np.mean(np.sqrt(dx**2 + dy**2))   # average step length

    # ── Hessian construction ──────────────────────────────────────────────────
    # kappa_i  ≈  (x_{i-1} - 2x_i + x_{i+1}) / ds²
    # In terms of alpha:
    #   x_i = rx_i + alpha_i * nx_i
    # so the 2nd diff of x w.r.t. alpha gives a band-diagonal matrix A (x-part)
    # and similarly for y.  H = (A_x^T A_x + A_y^T A_y) / ds^4

    # Build cyclic 2nd-difference operator  D ∈ R^{N×N}
    D = np.zeros((n, n))
    for i in range(n):
        D[i, (i - 1) % n] =  1.0
        D[i,  i          ] = -2.0
        D[i, (i + 1) % n] =  1.0
    D /= ds**2

    Nx = np.diag(normvecs[:, 0])
    Ny = np.diag(normvecs[:, 1])

    Ax = D @ Nx
    Ay = D @ Ny

    H  = 2.0 * (Ax.T @ Ax + Ay.T @ Ay)

    # Linear term from the reference line
    rx = refline[:, 0]
    ry = refline[:, 1]
    f  = 2.0 * (Ax.T @ (D @ rx) + Ay.T @ (D @ ry))

    # ── Bound constraints ─────────────────────────────────────────────────────
    # -w_right ≤ alpha ≤ w_left
    lb = -w_right
    ub =  w_left

    return H, f, lb, ub


def solve_qp_quadprog(H, f, lb, ub):
    """Solve the QP with quadprog (inequality form)."""
    n = len(f)
    # quadprog minimises  0.5 x^T G x - a^T x
    # subject to  C^T x >= b
    # We have:  min  0.5 x^T H x + f^T x
    #           -x <= -lb  →  x >= lb
    #            x <=  ub  →  -x >= -ub
    G  = H + np.eye(n) * 1e-8    # small regularisation for PD
    a  = -f
    # Stack: x >= lb  and  -x >= -ub
    C  = np.vstack([ np.eye(n), -np.eye(n)]).T
    b  = np.concatenate([lb, -ub])
    try:
        sol, _, _, _, _, _ = quadprog.solve_qp(G, a, C, b)
        return sol
    except Exception as e:
        warnings.warn(f"quadprog failed ({e}), using clipped zero.")
        return np.clip(np.zeros(n), lb, ub)


def solve_qp_osqp(H, f, lb, ub):
    """Solve the QP with OSQP."""
    import osqp, scipy.sparse as sps
    n = len(f)
    P = sps.csc_matrix(H + np.eye(n) * 1e-8)
    q = f
    A = sps.eye(n, format="csc")
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub,
               verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
    res = prob.solve()
    if res.info.status == "solved":
        return res.x
    warnings.warn(f"OSQP status: {res.info.status}")
    return np.clip(np.zeros(n), lb, ub)


def min_curvature_optimisation(refline:  np.ndarray,
                               normvecs: np.ndarray,
                               w_right:  np.ndarray,
                               w_left:   np.ndarray,
                               n_iter:   int = 3,
                               kappa_bound: float = None):
    """
    Iterative minimum-curvature QP  (Heilmeier 2020).

    Each iteration re-linearises the curvature around the previous solution
    to reduce linearisation error in sharp corners.

    Returns alpha (lateral offsets) and the optimised raceline.
    """
    if _QP_BACKEND is None:
        warnings.warn("No QP solver found (install quadprog or osqp). "
                      "Falling back to centreline.")
        return np.zeros(len(refline)), refline.copy()

    alpha = np.zeros(len(refline))

    for it in range(n_iter):
        raceline = refline + alpha[:, None] * normvecs
        H, f, lb, ub = _build_qp_matrices(raceline, normvecs, w_right, w_left, kappa_bound)

        if _QP_BACKEND == "quadprog":
            delta_alpha = solve_qp_quadprog(H, f, lb, ub)
        else:
            delta_alpha = solve_qp_osqp(H, f, lb, ub)

        alpha_new = alpha + delta_alpha
        alpha     = np.clip(alpha_new, -w_right, w_left)

        # convergence
        change = np.max(np.abs(delta_alpha))
        print(f"  [QP iter {it+1}/{n_iter}] max Δα = {change:.4f} m")
        if change < 1e-4:
            print("  Converged.")
            break

    raceline = refline + alpha[:, None] * normvecs
    return alpha, raceline


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CURVATURE & HEADING COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_geometry(raceline: np.ndarray):
    """
    Compute arc-length, heading (psi), and curvature (kappa) of the raceline.
    Uses Menger's formula for curvature at each triplet of consecutive points.
    """
    n    = len(raceline)
    x, y = raceline[:, 0], raceline[:, 1]

    # Arc-length
    dx   = np.diff(np.append(x, x[0]))
    dy   = np.diff(np.append(y, y[0]))
    ds   = np.sqrt(dx**2 + dy**2)
    s    = np.concatenate([[0], np.cumsum(ds[:-1])])

    # Heading (tangent angle)
    psi  = np.arctan2(dy, dx)
    # wrap
    psi  = (psi + np.pi) % (2 * np.pi) - np.pi

    # Curvature via cross product of successive tangent vectors
    kappa = np.zeros(n)
    for i in range(n):
        p1 = raceline[(i - 1) % n]
        p2 = raceline[i]
        p3 = raceline[(i + 1) % n]
        d1 = p2 - p1;  d2 = p3 - p2
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        norm1 = np.linalg.norm(d1);  norm2 = np.linalg.norm(d2)
        if norm1 * norm2 < 1e-12:
            kappa[i] = 0.0
        else:
            # Menger curvature
            area2  = abs(cross)
            denom  = norm1 * norm2 * np.linalg.norm(p3 - p1)
            kappa[i] = 2.0 * area2 / (denom + 1e-12) * np.sign(cross)

    return s, psi, kappa, ds


# ──────────────────────────────────────────────────────────────────────────────
# 5.  VELOCITY PROFILE  (forward–backward solver)
# ──────────────────────────────────────────────────────────────────────────────

class VehicleParams:
    """F1TENTH / small race car default parameters – tune for your vehicle."""
    v_max_mps    = 8.0    # maximum speed  [m/s]
    v_min_mps    = 0.5    # minimum speed  [m/s]
    ax_max_mps2  = 6.0    # max longitudinal acceleration  [m/s²]
    ax_min_mps2  = -8.0   # max braking deceleration      [m/s²]  (negative)
    ay_max_mps2  = 7.0    # max lateral acceleration       [m/s²]
    mass_kg      = 3.5    # vehicle mass  [kg]
    mu           = 0.8    # friction coefficient


def velocity_profile(kappa: np.ndarray,
                     ds:    np.ndarray,
                     vp:    VehicleParams) -> tuple:
    """
    Forward–backward integration velocity profile.
    Step 1: cornering-limited speed at each point  v_max_i = sqrt(ay_max / |kappa_i|)
    Step 2: forward pass  – acceleration limited
    Step 3: backward pass – braking limited
    Returns (vx, ax) arrays.
    """
    n   = len(kappa)
    eps = 1e-6

    # Cornering limit
    v_corner = np.where(np.abs(kappa) > eps,
                        np.sqrt(vp.ay_max_mps2 / (np.abs(kappa) + eps)),
                        vp.v_max_mps)
    v_corner = np.clip(v_corner, vp.v_min_mps, vp.v_max_mps)

    # Forward pass
    vf = np.zeros(n)
    vf[0] = vp.v_min_mps
    for i in range(1, n):
        d = ds[(i - 1) % n]
        v_possible = math.sqrt(max(vf[i-1]**2 + 2.0 * vp.ax_max_mps2 * d, 0.0))
        vf[i] = min(v_possible, v_corner[i])

    # Backward pass
    vb = np.zeros(n)
    vb[-1] = vf[-1]
    for i in range(n - 2, -1, -1):
        d = ds[i]
        v_possible = math.sqrt(max(vb[i+1]**2 - 2.0 * vp.ax_min_mps2 * d, 0.0))
        vb[i] = min(v_possible, v_corner[i])

    vx = np.minimum(vf, vb)
    vx = np.clip(vx, vp.v_min_mps, vp.v_max_mps)

    # Acceleration
    ax = np.zeros(n)
    for i in range(n):
        d   = ds[i]
        dv  = vx[(i + 1) % n] - vx[i]
        ax[i] = (vx[(i + 1) % n]**2 - vx[i]**2) / (2.0 * d + 1e-9)
    ax = np.clip(ax, vp.ax_min_mps2, vp.ax_max_mps2)

    return vx, ax


# ──────────────────────────────────────────────────────────────────────────────
# 6.  OUTPUT WRITING
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(path: str,
              raceline: np.ndarray,
              w_right:  np.ndarray,
              w_left:   np.ndarray,
              psi:      np.ndarray,
              kappa:    np.ndarray,
              vx:       np.ndarray,
              ax:       np.ndarray):
    header = "x_m,y_m,w_tr_right_m,w_tr_left_m,psi_rad,kappa_radpm,vx_mps,ax_mps2"
    data   = np.column_stack([raceline, w_right, w_left, psi, kappa, vx, ax])
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"[output] Raceline CSV written: {path}  ({len(raceline)} points)")


def visualise(map_data:   dict,
              centreline: np.ndarray,
              raceline:   np.ndarray,
              vx:         np.ndarray,
              out_path:   str):
    """Render occupancy grid + centreline + colour-coded raceline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    res   = map_data["resolution"]
    grid  = map_data["grid"]
    ox    = map_data["origin_x"]
    oy    = map_data["origin_y"]

    # Convert world coords → image pixel
    def w2px(xy):
        px = (xy[:, 0] - ox) / res
        py = (-xy[:, 1] - 0) / res    # flip y back
        # shift by image origin row (image origin at bottom-left in ROS)
        py = grid.shape[0] - py
        return px, py

    for ax_idx, (ax_obj, title) in enumerate(zip(axes, ["Map Overview", "Velocity Profile"])):
        ax_obj.imshow(grid, cmap="gray", origin="upper",
                      extent=[ox, ox + grid.shape[1]*res,
                               oy - grid.shape[0]*res, oy])
        ax_obj.set_aspect("equal")
        ax_obj.set_xlabel("x [m]")
        ax_obj.set_ylabel("y [m]")
        ax_obj.set_title(title)

        if ax_idx == 0:
            # Centreline
            ax_obj.plot(centreline[:, 0], centreline[:, 1],
                        "c--", lw=1, alpha=0.6, label="Centreline")
            # Raceline
            ax_obj.plot(raceline[:, 0], raceline[:, 1],
                        "r-", lw=2, label="Min-Curvature Raceline")
            ax_obj.legend(loc="upper right", fontsize=8)
        else:
            # Colour-coded by velocity
            from matplotlib.collections import LineCollection
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            pts_seg = np.array([raceline[:-1], raceline[1:]]).transpose(1, 0, 2)
            # close loop
            pts_seg = np.vstack([pts_seg,
                                  [[raceline[-1], raceline[0]]]])
            norm   = Normalize(vmin=vx.min(), vmax=vx.max())
            colors = plt.cm.RdYlGn(norm(vx))
            lc     = LineCollection(pts_seg, colors=colors, linewidth=2)
            ax_obj.add_collection(lc)
            sm = ScalarMappable(cmap="RdYlGn", norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax_obj, label="Speed [m/s]", fraction=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[output] Visualisation saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  NORMAL VECTOR COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_normvecs(pts: np.ndarray) -> np.ndarray:
    """Unit normal vectors (pointing left of direction of travel) for each point."""
    n    = len(pts)
    tang = np.zeros((n, 2))
    for i in range(n):
        d = pts[(i + 1) % n] - pts[(i - 1) % n]
        l = np.linalg.norm(d)
        tang[i] = d / l if l > 1e-9 else np.array([1.0, 0.0])
    # Left normal: rotate tangent 90° CCW
    normvecs = np.column_stack([-tang[:, 1], tang[:, 0]])
    return normvecs


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(pbstream_path:   str = None,
                 pgm_path:        str = None,
                 yaml_path:       str = None,
                 output_dir:      str = ".",
                 step_m:          float = 0.1,
                 safety_margin_m: float = 0.1,
                 n_iter:          int   = 3,
                 resolution:      float = None,
                 vp:              VehicleParams = None,
                 dilation_px:     int   = 3):

    if vp is None:
        vp = VehicleParams()

    os.makedirs(output_dir, exist_ok=True)

    # ── Load map ──────────────────────────────────────────────────────────────
    print("\n[1/7] Loading map...")
    if pgm_path:
        map_data = _load_pgm_yaml(pgm_path, yaml_path)
    elif pbstream_path:
        map_data = load_map_from_pbstream(pbstream_path)
    else:
        raise ValueError("Provide either --pbstream or --pgm")

    if resolution:
        map_data["resolution"] = resolution
        print(f"      Resolution overridden to {resolution} m/px")
    res = map_data["resolution"]
    print(f"      Grid: {map_data['grid'].shape}  @  {res} m/px")

    # ── Pre-process ───────────────────────────────────────────────────────────
    print("[2/7] Pre-processing occupancy grid...")
    clean_mask = preprocess_map(map_data, dilation_px=dilation_px)

    # ── Extract centreline ────────────────────────────────────────────────────
    print("[3/7] Extracting centreline via skeletonisation...")
    centreline_px_raw, skel_img = extract_centreline(clean_mask, res)

    # Convert px centreline back to pixel coords for width computation
    # centreline_px_raw is in world metres; undo the conversion
    centreline_px = centreline_px_raw.copy()
    centreline_px[:, 0] =  centreline_px_raw[:, 0] / res
    centreline_px[:, 1] = -centreline_px_raw[:, 1] / res

    print(f"      {len(centreline_px)} skeleton points found")

    # ── Track widths ──────────────────────────────────────────────────────────
    print("[4/7] Computing track widths...")
    w_right_raw, w_left_raw = compute_track_widths(
        centreline_px, clean_mask, res, safety_margin_m)

    # ── Smooth & resample ─────────────────────────────────────────────────────
    print("[5/7] Smoothing & resampling centreline...")
    refline, _, _ = smooth_and_resample(centreline_px_raw, step_m=step_m)
    print(f"      Resampled to {len(refline)} points at {step_m} m intervals")

    # Interpolate widths to resampled grid
    t_raw = np.linspace(0, 1, len(centreline_px_raw))
    t_new = np.linspace(0, 1, len(refline))
    w_right = np.interp(t_new, t_raw, w_right_raw)
    w_left  = np.interp(t_new, t_raw, w_left_raw)

    # Normal vectors for the refline
    normvecs = compute_normvecs(refline)

    # ── Minimum-curvature QP ──────────────────────────────────────────────────
    print(f"[6/7] Minimum-curvature QP optimisation (backend={_QP_BACKEND}, "
          f"{n_iter} iterations)...")
    alpha, raceline = min_curvature_optimisation(
        refline, normvecs, w_right, w_left, n_iter=n_iter)

    # Adjust widths relative to optimised raceline
    w_right_rl = w_right - alpha
    w_left_rl  = w_left  + alpha

    # ── Geometry & velocity profile ───────────────────────────────────────────
    print("[7/7] Computing curvature and velocity profile...")
    s, psi, kappa, ds = compute_geometry(raceline)
    vx, ax = velocity_profile(kappa, ds, vp)

    lap_time = np.sum(ds / (vx + 1e-9))
    print(f"      Estimated lap time: {lap_time:.2f} s")
    print(f"      Max speed: {vx.max():.2f} m/s   Min speed: {vx.min():.2f} m/s")

    # ── Write outputs ─────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "raceline.csv")
    png_path = os.path.join(output_dir, "raceline.png")

    write_csv(csv_path, raceline, w_right_rl, w_left_rl, psi, kappa, vx, ax)
    visualise(map_data, refline, raceline, vx, png_path)

    return csv_path, png_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Minimum-curvature raceline generator from Cartographer pbstream")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pbstream",  help="Path to .pbstream file")
    grp.add_argument("--pgm",       help="Path to pre-exported .pgm / .png map image")

    parser.add_argument("--yaml",        default=None,  help=".yaml metadata for the pgm map")
    parser.add_argument("--output",      default="output", help="Output directory")
    parser.add_argument("--step",        type=float, default=0.1,  help="Raceline step size [m]")
    parser.add_argument("--safety",      type=float, default=0.1,  help="Safety margin from walls [m]")
    parser.add_argument("--iter",        type=int,   default=3,    help="QP iterations")
    parser.add_argument("--resolution",  type=float, default=None, help="Override map resolution [m/px]")
    parser.add_argument("--dilation",    type=int,   default=3,    help="Wall dilation [px]")

    # Vehicle params
    parser.add_argument("--v_max",   type=float, default=8.0,  help="Max speed [m/s]")
    parser.add_argument("--ax_max",  type=float, default=6.0,  help="Max accel [m/s²]")
    parser.add_argument("--ax_min",  type=float, default=-8.0, help="Max braking [m/s²]")
    parser.add_argument("--ay_max",  type=float, default=7.0,  help="Max lat accel [m/s²]")

    args = parser.parse_args()

    vp = VehicleParams()
    vp.v_max_mps   = args.v_max
    vp.ax_max_mps2 = args.ax_max
    vp.ax_min_mps2 = args.ax_min
    vp.ay_max_mps2 = args.ay_max

    run_pipeline(
        pbstream_path   = args.pbstream,
        pgm_path        = args.pgm,
        yaml_path       = args.yaml,
        output_dir      = args.output,
        step_m          = args.step,
        safety_margin_m = args.safety,
        n_iter          = args.iter,
        resolution      = args.resolution,
        vp              = vp,
        dilation_px     = args.dilation,
    )


if __name__ == "__main__":
    main()
