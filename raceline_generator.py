"""
Minimum Curvature Raceline Generator
=====================================
Fixed for annular (ring-shaped) tracks like AutoDRIVE RoboRacer.
 
Input  : .pgm + .yaml map (exported from Cartographer pbstream)
Output : raceline.csv + raceline.png
"""
 
import os, sys, argparse, math, warnings
import numpy as np
import cv2
from scipy import ndimage, interpolate
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
try:
    import quadprog
    _QP_BACKEND = "quadprog"
except ImportError:
    try:
        import osqp, scipy.sparse as sp
        _QP_BACKEND = "osqp"
    except ImportError:
        _QP_BACKEND = None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. MAP LOADING
# ─────────────────────────────────────────────────────────────────────────────
 
def load_map(pgm_path, yaml_path=None):
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {pgm_path}")
 
    resolution = 0.05
    origin_x   = 0.0
    origin_y   = 0.0
 
    if yaml_path and os.path.isfile(yaml_path):
        import yaml
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        resolution = float(meta.get("resolution", resolution))
        origin     = meta.get("origin", [0, 0, 0])
        origin_x, origin_y = float(origin[0]), float(origin[1])
 
    return {"grid": img, "resolution": resolution,
            "origin_x": origin_x, "origin_y": origin_y}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. TRACK MASK EXTRACTION  ←  THE KEY FIX
#
# For a ring-shaped track the occupancy grid has:
#   ~254  free   (white) = drivable track surface + areas outside outer wall
#   ~205  unknown (grey) = areas outside the scanned region
#   ~0    occupied (black) = the actual walls
#
# Problem: standard "free = drivable" gives us the entire white blob including
#          the inner island region which looks free but is surrounded by a wall.
#
# Fix: extract the RING = largest free blob MINUS any enclosed sub-blobs.
#      We detect the track ring by:
#      1. Threshold → free mask
#      2. Remove the largest connected component (the outer open area if any)
#         OR keep only the second-largest ... 
#      Actually the robust approach:
#      1. Free mask
#      2. Find all connected components of the FREE region
#      3. The TRACK is the free component that:
#         - Contains wall pixels (occupied) on BOTH its inner and outer boundary
#         i.e. it is enclosed on both sides
#      Simpler robust approach used here:
#      - Erode the free mask heavily → removes thin track corridors, keeps blobs
#      - The track corridor disappears; inner/outer regions remain
#      - Subtract eroded from original → just the corridor ring
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_track_mask(map_data, safety_margin_m=0.1):
    img = map_data["grid"]
    res = map_data["resolution"]
 
    # Step 1: binary free mask  (white pixels = free/drivable)
    free = (img > 180).astype(np.uint8) * 255
 
    # Step 2: find connected components of free space
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
 
    print(f"  Free-space connected components: {n_labels - 1}")
    for i in range(1, min(n_labels, 6)):
        print(f"    CC {i}: area={stats[i, cv2.CC_STAT_AREA]} px")
 
    if n_labels < 2:
        raise RuntimeError("No free space found in map. Check map thresholds.")
 
    # Sort components by area descending (skip label 0 = background)
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n_labels)]
    areas.sort(reverse=True)
 
    if len(areas) == 1:
        # Only one free blob — treat it as the track directly
        track_label = areas[0][1]
        track_mask  = ((labels == track_label).astype(np.uint8)) * 255
        print("  Single free blob — using as track directly")
    else:
        # Multiple free blobs.
        # Strategy: the track corridor is the blob that when eroded heavily
        # disappears (it's narrow), whereas inner/outer open areas survive erosion.
        # We try each candidate and pick the one whose eroded version is smallest.
        track_mask = None
        best_score = -1
 
        # Try the top-3 largest blobs
        candidates = [idx for _, idx in areas[:5]]
 
        for ci in candidates:
            blob = ((labels == ci).astype(np.uint8)) * 255
            area = stats[ci, cv2.CC_STAT_AREA]
 
            # Estimate track width: erode by track_width/2 and see how much survives
            erode_px = max(3, int(0.3 / res))   # 30 cm erosion
            kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (erode_px*2+1, erode_px*2+1))
            eroded   = cv2.erode(blob, kernel, iterations=1)
            survival = np.count_nonzero(eroded) / (area + 1)
 
            # Check if it's ring-shaped: after filling holes, area increases a lot
            filled = ndimage.binary_fill_holes(blob > 0).astype(np.uint8) * 255
            fill_ratio = np.count_nonzero(filled) / (area + 1)
 
            print(f"  CC {ci}: area={area}px  erosion_survival={survival:.3f}  "
                  f"fill_ratio={fill_ratio:.2f}")
 
            # A ring/corridor has high fill_ratio (filling the hole adds area)
            # AND medium erosion survival (corridor is not too thin but not huge)
            score = fill_ratio  # higher = more ring-like
            if score > best_score:
                best_score  = score
                track_mask  = blob
 
        if track_mask is None:
            track_mask = ((labels == areas[0][1]).astype(np.uint8)) * 255
 
    # Step 3: morphological closing to fill small gaps in the track
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, k2, iterations=2)
 
    # Step 4: safety margin — erode slightly so centreline stays away from walls
    margin_px = max(1, int(safety_margin_m / res))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (margin_px*2+1, margin_px*2+1))
    track_mask = cv2.erode(track_mask, k3, iterations=1)
 
    return track_mask
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. CENTRELINE VIA SKELETONISATION OF THE RING
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_centreline(track_mask, resolution):
    from skimage.morphology import skeletonize
 
    binary = (track_mask > 127).astype(np.uint8)
    skel   = skeletonize(binary).astype(np.uint8) * 255
 
    ys, xs = np.where(skel > 0)
    if len(xs) < 10:
        raise RuntimeError(
            "Skeleton too sparse — track mask may be wrong.\n"
            "Try --safety 0.05 or check raceline.png map panel.")
 
    pts = np.column_stack([xs, ys]).astype(float)
    pts_ordered = _order_points_nn(pts)
 
    # Convert pixel → world coords
    pts_m = pts_ordered.copy()
    pts_m[:, 0] =  pts_ordered[:, 0] * resolution
    pts_m[:, 1] = -pts_ordered[:, 1] * resolution   # flip y (image y is down)
 
    return pts_m, skel
 
 
def _order_points_nn(pts):
    """Order unordered 2-D points along a closed loop via nearest-neighbour walk."""
    n    = len(pts)
    tree = KDTree(pts)
    visited = np.zeros(n, dtype=bool)
    order   = [0]
    visited[0] = True
 
    for _ in range(n - 1):
        current = order[-1]
        _, idxs = tree.query(pts[current], k=min(20, n))
        for idx in idxs[1:]:
            if not visited[idx]:
                order.append(idx)
                visited[idx] = True
                break
 
    return pts[order]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. TRACK WIDTHS
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_track_widths(centreline_px, mask, resolution, safety_margin_m=0.1):
    n  = len(centreline_px)
    h, w = mask.shape
    w_right = np.zeros(n)
    w_left  = np.zeros(n)
 
    for i in range(n):
        p_prev = centreline_px[(i - 1) % n]
        p_next = centreline_px[(i + 1) % n]
        tang   = p_next - p_prev
        tlen   = np.linalg.norm(tang)
        if tlen < 1e-9:
            w_right[i] = safety_margin_m
            w_left[i]  = safety_margin_m
            continue
        tang /= tlen
        nr = np.array([ tang[1], -tang[0]])   # right normal
        nl = np.array([-tang[1],  tang[0]])   # left normal
 
        cx, cy = centreline_px[i]
        for normal, store in [(nr, w_right), (nl, w_left)]:
            dist = 0.0
            for step in range(1, 600):
                nx = int(round(cx + normal[0] * step))
                ny = int(round(cy + normal[1] * step))
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    dist = step * resolution; break
                if mask[ny, nx] < 127:
                    dist = step * resolution; break
            else:
                dist = 600 * resolution
            store[i] = max(dist - safety_margin_m, 0.05)
 
    return w_right, w_left
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. SMOOTH & RESAMPLE
# ─────────────────────────────────────────────────────────────────────────────
 
def smooth_and_resample(pts_m, step_m=0.1, smooth_s=None):
    pts_closed = np.vstack([pts_m, pts_m[0]])
    x, y = pts_closed[:, 0], pts_closed[:, 1]
    dx = np.diff(x); dy = np.diff(y)
    seg = np.sqrt(dx**2 + dy**2)
    t   = np.concatenate([[0], np.cumsum(seg)])
    total_len = t[-1]
 
    if smooth_s is None:
        smooth_s = total_len * 0.01
 
    tck, _ = interpolate.splprep([x, y], u=t, s=smooth_s, per=True, k=5)
    n_pts  = max(50, int(total_len / step_m))
    u_new  = np.linspace(0, total_len, n_pts, endpoint=False)
    xs, ys = interpolate.splev(u_new, tck)
 
    return np.column_stack([xs, ys]), tck, total_len
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. NORMAL VECTORS
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_normvecs(pts):
    n = len(pts)
    tang = np.zeros((n, 2))
    for i in range(n):
        d = pts[(i + 1) % n] - pts[(i - 1) % n]
        l = np.linalg.norm(d)
        tang[i] = d / l if l > 1e-9 else np.array([1.0, 0.0])
    return np.column_stack([-tang[:, 1], tang[:, 0]])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. MINIMUM CURVATURE QP
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_qp(refline, normvecs, w_right, w_left):
    n  = len(refline)
    dx = np.diff(np.append(refline[:, 0], refline[0, 0]))
    dy = np.diff(np.append(refline[:, 1], refline[1, 1]))
    ds = np.mean(np.sqrt(dx**2 + dy**2))
 
    D = np.zeros((n, n))
    for i in range(n):
        D[i, (i-1)%n] =  1.0
        D[i,  i      ] = -2.0
        D[i, (i+1)%n] =  1.0
    D /= ds**2
 
    Nx = np.diag(normvecs[:, 0])
    Ny = np.diag(normvecs[:, 1])
    Ax = D @ Nx;  Ay = D @ Ny
    H  = 2.0 * (Ax.T @ Ax + Ay.T @ Ay)
    rx = refline[:, 0];  ry = refline[:, 1]
    f  = 2.0 * (Ax.T @ (D @ rx) + Ay.T @ (D @ ry))
    return H, f, -w_right, w_left
 
 
def _solve_qp(H, f, lb, ub):
    n = len(f)
    if _QP_BACKEND == "quadprog":
        G = H + np.eye(n) * 1e-8
        a = -f
        C = np.vstack([np.eye(n), -np.eye(n)]).T
        b = np.concatenate([lb, -ub])
        try:
            sol, *_ = quadprog.solve_qp(G, a, C, b)
            return sol
        except Exception as e:
            warnings.warn(f"quadprog failed: {e}")
            return np.clip(np.zeros(n), lb, ub)
    elif _QP_BACKEND == "osqp":
        import osqp, scipy.sparse as sps
        P = sps.csc_matrix(H + np.eye(n)*1e-8)
        A = sps.eye(n, format="csc")
        prob = osqp.OSQP()
        prob.setup(P, f, A, lb, ub, verbose=False, eps_abs=1e-6, max_iter=10000)
        res = prob.solve()
        return res.x if res.info.status == "solved" else np.clip(np.zeros(n), lb, ub)
    else:
        warnings.warn("No QP solver — using centreline")
        return np.clip(np.zeros(n), lb, ub)
 
 
def min_curvature_optimisation(refline, normvecs, w_right, w_left, n_iter=3):
    alpha = np.zeros(len(refline))
    for it in range(n_iter):
        raceline = refline + alpha[:, None] * normvecs
        H, f, lb, ub = _build_qp(raceline, normvecs, w_right, w_left)
        delta = _solve_qp(H, f, lb, ub)
        alpha = np.clip(alpha + delta, -w_right, w_left)
        change = np.max(np.abs(delta))
        print(f"  [QP iter {it+1}/{n_iter}] max Δα = {change:.4f} m")
        if change < 1e-4:
            print("  Converged.")
            break
    return alpha, refline + alpha[:, None] * normvecs
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_geometry(raceline):
    n  = len(raceline)
    x, y = raceline[:, 0], raceline[:, 1]
    dx = np.diff(np.append(x, x[0]))
    dy = np.diff(np.append(y, y[0]))
    ds = np.sqrt(dx**2 + dy**2)
    s  = np.concatenate([[0], np.cumsum(ds[:-1])])
    psi = np.arctan2(dy, dx)
    psi = (psi + np.pi) % (2*np.pi) - np.pi
 
    kappa = np.zeros(n)
    for i in range(n):
        p1 = raceline[(i-1)%n];  p2 = raceline[i];  p3 = raceline[(i+1)%n]
        d1 = p2-p1;  d2 = p3-p2
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        denom = np.linalg.norm(d1) * np.linalg.norm(d2) * np.linalg.norm(p3-p1)
        kappa[i] = 2.0 * abs(cross) / (denom + 1e-12) * np.sign(cross)
 
    return s, psi, kappa, ds
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. VELOCITY PROFILE
# ─────────────────────────────────────────────────────────────────────────────
 
class VehicleParams:
    v_max_mps   = 8.0
    v_min_mps   = 0.5
    ax_max_mps2 = 6.0
    ax_min_mps2 = -8.0
    ay_max_mps2 = 7.0
 
 
def velocity_profile(kappa, ds, vp):
    n   = len(kappa)
    eps = 1e-6
    v_corner = np.where(np.abs(kappa) > eps,
                        np.sqrt(vp.ay_max_mps2 / (np.abs(kappa) + eps)),
                        vp.v_max_mps)
    v_corner = np.clip(v_corner, vp.v_min_mps, vp.v_max_mps)
 
    vf = np.zeros(n);  vf[0] = vp.v_min_mps
    for i in range(1, n):
        d = ds[(i-1)%n]
        vf[i] = min(math.sqrt(max(vf[i-1]**2 + 2*vp.ax_max_mps2*d, 0)), v_corner[i])
 
    vb = np.zeros(n);  vb[-1] = vf[-1]
    for i in range(n-2, -1, -1):
        vb[i] = min(math.sqrt(max(vb[i+1]**2 - 2*vp.ax_min_mps2*ds[i], 0)), v_corner[i])
 
    vx = np.clip(np.minimum(vf, vb), vp.v_min_mps, vp.v_max_mps)
    ax = np.array([(vx[(i+1)%n]**2 - vx[i]**2) / (2*ds[i]+1e-9) for i in range(n)])
    ax = np.clip(ax, vp.ax_min_mps2, vp.ax_max_mps2)
    return vx, ax
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 10. OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
 
def write_csv(path, raceline, w_right, w_left, psi, kappa, vx, ax):
    header = "x_m,y_m,w_tr_right_m,w_tr_left_m,psi_rad,kappa_radpm,vx_mps,ax_mps2"
    np.savetxt(path, np.column_stack([raceline, w_right, w_left, psi, kappa, vx, ax]),
               delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"[output] CSV: {path}  ({len(raceline)} points)")
 
 
def visualise(map_data, track_mask, centreline_m, raceline_m, vx, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    res = map_data["resolution"]
    ox  = map_data["origin_x"]
    oy  = map_data["origin_y"]
    grid = map_data["grid"]
    H    = grid.shape[0]
 
    # World-coord extent for imshow
    ext = [ox, ox + grid.shape[1]*res, oy - H*res, oy]
 
    for ax_i, (ax_obj, title) in enumerate(zip(axes, ["Map + Raceline", "Velocity Profile"])):
        ax_obj.imshow(grid, cmap="gray", origin="upper", extent=ext)
        ax_obj.set_aspect("equal")
        ax_obj.set_xlabel("x [m]"); ax_obj.set_ylabel("y [m]")
        ax_obj.set_title(title)
 
        if ax_i == 0:
            # Apply origin offset to convert pixel-space to world-space
            def px2world(pts_m):
                # pts_m are already in "pixel world" (pixel * res, y flipped)
                # need to add origin offset
                w = pts_m.copy()
                w[:, 0] += ox
                w[:, 1] += oy  # oy is negative, so this shifts correctly
                return w
 
            cl_w = px2world(centreline_m)
            rl_w = px2world(raceline_m)
 
            ax_obj.plot(cl_w[:, 0], cl_w[:, 1], "c--", lw=1, alpha=0.6, label="Centreline")
            ax_obj.plot(rl_w[:, 0], rl_w[:, 1], "r-",  lw=2.5, label="Min-Curvature Raceline")
            ax_obj.legend(loc="upper right", fontsize=8)
        else:
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
            rl_w = raceline_m.copy()
            rl_w[:, 0] += ox
            rl_w[:, 1] += oy
 
            pts_seg = np.array([rl_w[:-1], rl_w[1:]]).transpose(1, 0, 2)
            pts_seg = np.vstack([pts_seg, [[rl_w[-1], rl_w[0]]]])
            norm    = Normalize(vmin=vx.min(), vmax=vx.max())
            lc = LineCollection(pts_seg, cmap="RdYlGn", norm=norm, linewidth=2.5)
            lc.set_array(vx)
            ax_obj.add_collection(lc)
            plt.colorbar(lc, ax=ax_obj, label="Speed [m/s]", fraction=0.04)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[output] PNG: {out_path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
 
def run_pipeline(pgm_path, yaml_path=None, output_dir=".", step_m=0.1,
                 safety_margin_m=0.1, n_iter=3, vp=None):
    if vp is None:
        vp = VehicleParams()
    os.makedirs(output_dir, exist_ok=True)
 
    print("\n[1/7] Loading map...")
    map_data = load_map(pgm_path, yaml_path)
    res = map_data["resolution"]
    print(f"      Grid: {map_data['grid'].shape}  @  {res} m/px")
    print(f"      Origin: ({map_data['origin_x']:.3f}, {map_data['origin_y']:.3f})")
 
    print("[2/7] Extracting track mask (ring-aware)...")
    track_mask = extract_track_mask(map_data, safety_margin_m)
    # Save debug mask
    cv2.imwrite(os.path.join(output_dir, "debug_track_mask.png"), track_mask)
 
    print("[3/7] Skeletonising track corridor...")
    centreline_m, skel = extract_centreline(track_mask, res)
    print(f"      {len(centreline_m)} skeleton points")
 
    # Pixel coords for width computation
    centreline_px = centreline_m.copy()
    centreline_px[:, 0] =  centreline_m[:, 0] / res
    centreline_px[:, 1] = -centreline_m[:, 1] / res
 
    print("[4/7] Computing track widths...")
    w_right_raw, w_left_raw = compute_track_widths(
        centreline_px, track_mask, res, safety_margin_m)
 
    print("[5/7] Smoothing & resampling...")
    refline, _, _ = smooth_and_resample(centreline_m, step_m=step_m)
    print(f"      {len(refline)} points @ {step_m} m")
 
    t_raw = np.linspace(0, 1, len(centreline_m))
    t_new = np.linspace(0, 1, len(refline))
    w_right = np.interp(t_new, t_raw, w_right_raw)
    w_left  = np.interp(t_new, t_raw, w_left_raw)
    normvecs = compute_normvecs(refline)
 
    print(f"[6/7] Min-curvature QP (backend={_QP_BACKEND}, {n_iter} iter)...")
    alpha, raceline = min_curvature_optimisation(
        refline, normvecs, w_right, w_left, n_iter)
 
    w_right_rl = w_right - alpha
    w_left_rl  = w_left  + alpha
 
    print("[7/7] Geometry + velocity profile...")
    s, psi, kappa, ds = compute_geometry(raceline)
    vx, ax = velocity_profile(kappa, ds, vp)
 
    lap = np.sum(ds / (vx + 1e-9))
    print(f"      Lap time: {lap:.2f} s | "
          f"Speed: {vx.min():.2f}–{vx.max():.2f} m/s")
 
    write_csv(os.path.join(output_dir, "raceline.csv"),
              raceline, w_right_rl, w_left_rl, psi, kappa, vx, ax)
    visualise(map_data, track_mask, refline, raceline, vx,
              os.path.join(output_dir, "raceline.png"))
 
    return raceline
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgm",    required=True)
    parser.add_argument("--yaml",   default=None)
    parser.add_argument("--output", default="output")
    parser.add_argument("--step",   type=float, default=0.1)
    parser.add_argument("--safety", type=float, default=0.1)
    parser.add_argument("--iter",   type=int,   default=3)
    parser.add_argument("--v_max",  type=float, default=8.0)
    parser.add_argument("--ax_max", type=float, default=6.0)
    parser.add_argument("--ax_min", type=float, default=-8.0)
    parser.add_argument("--ay_max", type=float, default=7.0)
    args = parser.parse_args()
 
    vp = VehicleParams()
    vp.v_max_mps = args.v_max;  vp.ax_max_mps2 = args.ax_max
    vp.ax_min_mps2 = args.ax_min;  vp.ay_max_mps2 = args.ay_max
 
    run_pipeline(args.pgm, args.yaml, args.output,
                 args.step, args.safety, args.iter, vp)
 
if __name__ == "__main__":
    main()
 