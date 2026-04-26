#!/usr/bin/env python3
"""
Minimum LAP TIME Raceline Generator
=====================================
Switches from minimum-curvature to minimum lap time optimisation.

Algorithm:
  1. Extract track corridor + centreline (same as before)
  2. Smooth & resample centreline
  3. Compute track widths
  4. MINIMUM LAP TIME optimisation:
     - Iterative QP on lateral offsets (same structure as min-curvature)
     - BUT velocity profile uses a friction-circle (combined ax+ay) constraint
     - AND the QP objective weights curvature by local speed (high-speed
       sections penalise curvature MORE → wider arcs where it matters most)
  5. Forward-backward velocity integration with friction circle
  6. Iterate raceline ↔ velocity until convergence

Vehicle params (AutoDRIVE RoboRacer technical guide 2026):
  Wheelbase    : 0.3240 m
  Top speed    : 22.88 m/s
  Steering     : ±0.5236 rad
  Mass         : 3.906 kg
  ay_max       : 9.8 m/s²  (lateral tire extremum 1.0g)
  ax_max       : 8.0 m/s²  (longitudinal — conservative estimate)
  ax_min       : -10.0 m/s² (braking)
  mu (friction): 1.0  (combined friction circle radius = sqrt(ax²+ay²) ≤ mu*g)
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
    _QP = "quadprog"
except ImportError:
    try:
        import osqp
        _QP = "osqp"
    except ImportError:
        _QP = None

print(f"QP backend: {_QP}")


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle params — AutoDRIVE RoboRacer 2026
# ─────────────────────────────────────────────────────────────────────────────
class Vehicle:
    wheelbase    = 0.3240   # m
    v_max        = 22.88    # m/s
    v_min        = 0.5      # m/s
    ax_max       = 8.0      # m/s²  acceleration
    ax_min       = -10.0    # m/s²  braking
    ay_max       = 9.8      # m/s²  lateral (from tire model)
    mu           = 1.0      # friction coefficient
    g            = 9.81     # m/s²
    # Friction circle: sqrt((ax/ax_lim)² + (ay/ay_max)²) ≤ 1
    # Combined limit: a_total ≤ mu*g = 9.81 m/s²
    a_total_max  = 9.81     # m/s²  (mu * g)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAP
# ─────────────────────────────────────────────────────────────────────────────

def load_map(pgm_path, yaml_path=None):
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read: {pgm_path}")
    res = 0.05; ox = 0.0; oy = 0.0
    if yaml_path and os.path.isfile(yaml_path):
        import yaml
        with open(yaml_path) as f:
            m = yaml.safe_load(f)
        res = float(m.get("resolution", res))
        org = m.get("origin", [0,0,0])
        ox, oy = float(org[0]), float(org[1])
    H, W = img.shape
    return {"grid": img, "resolution": res,
            "origin_x": ox, "origin_y": oy, "H": H, "W": W}


def px_to_world(col, row, res, ox, oy, H):
    return col*res + ox, (H-1-row)*res + oy

def pts_px_to_world(pts, res, ox, oy, H):
    out = np.zeros_like(pts)
    out[:,0] = pts[:,0]*res + ox
    out[:,1] = (H-1-pts[:,1])*res + oy
    return out

def pts_world_to_px(pts, res, ox, oy, H):
    out = np.zeros_like(pts)
    out[:,0] = (pts[:,0]-ox)/res
    out[:,1] = H-1-(pts[:,1]-oy)/res
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRACK MASK
# ─────────────────────────────────────────────────────────────────────────────

def extract_track_mask(md, safety_m=0.1):
    img = md["grid"]; res = md["resolution"]
    free = (img > 180).astype(np.uint8)*255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    print(f"  Free CCs: {n-1}")
    areas = sorted([(stats[i,cv2.CC_STAT_AREA],i) for i in range(1,n)], reverse=True)
    if len(areas)==1:
        mask = ((labels==areas[0][1]).astype(np.uint8))*255
    else:
        best=-1; mask=None
        for _,ci in areas[:5]:
            blob=((labels==ci).astype(np.uint8))*255
            area=stats[ci,cv2.CC_STAT_AREA]
            filled=ndimage.binary_fill_holes(blob>0).astype(np.uint8)*255
            fr=np.count_nonzero(filled)/(area+1)
            print(f"    CC{ci}: area={area} fill_ratio={fr:.2f}")
            if fr>best: best=fr; mask=blob
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    mp=max(1,int(safety_m/res))
    k2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mp*2+1,mp*2+1))
    return cv2.erode(mask,k2,iterations=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CENTRELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_centreline_px(mask):
    from skimage.morphology import skeletonize
    skel = skeletonize((mask>127).astype(np.uint8)).astype(np.uint8)*255
    ys,xs = np.where(skel>0)
    pts = np.column_stack([xs,ys]).astype(float)
    return _order_nn(pts), skel

def _order_nn(pts):
    n=len(pts); tree=KDTree(pts)
    vis=np.zeros(n,bool); order=[0]; vis[0]=True
    for _ in range(n-1):
        _,idxs=tree.query(pts[order[-1]],k=min(20,n))
        for i in idxs[1:]:
            if not vis[i]: order.append(i); vis[i]=True; break
    return pts[order]


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRACK WIDTHS
# ─────────────────────────────────────────────────────────────────────────────

def track_widths(cl_px, mask, res, safety_m=0.1):
    n=len(cl_px); H,W=mask.shape
    wr=np.zeros(n); wl=np.zeros(n)
    for i in range(n):
        tang=cl_px[(i+1)%n]-cl_px[(i-1)%n]
        tl=np.linalg.norm(tang)
        if tl<1e-9: wr[i]=wl[i]=safety_m; continue
        tang/=tl
        nr=np.array([tang[1],-tang[0]])
        nl=np.array([-tang[1],tang[0]])
        cx,cy=cl_px[i]
        for nm,st in [(nr,wr),(nl,wl)]:
            for s in range(1,600):
                nx=int(round(cx+nm[0]*s)); ny=int(round(cy+nm[1]*s))
                if nx<0 or ny<0 or nx>=W or ny>=H:
                    st[i]=max(s*res-safety_m,0.05); break
                if mask[ny,nx]<127:
                    st[i]=max(s*res-safety_m,0.05); break
            else: st[i]=max(600*res-safety_m,0.05)
    return wr,wl


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMOOTH & RESAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def smooth_resample(pts, step_m=0.05, smooth_s=None):
    pc=np.vstack([pts,pts[0]])
    x,y=pc[:,0],pc[:,1]
    seg=np.sqrt(np.diff(x)**2+np.diff(y)**2)
    t=np.concatenate([[0],np.cumsum(seg)]); L=t[-1]
    if smooth_s is None: smooth_s=L*0.005
    tck,_=interpolate.splprep([x,y],u=t,s=smooth_s,per=True,k=5)
    n=max(100,int(L/step_m))
    u=np.linspace(0,L,n,endpoint=False)
    xs,ys=interpolate.splev(u,tck)
    return np.column_stack([xs,ys]), L


# ─────────────────────────────────────────────────────────────────────────────
# 6. NORMAL VECTORS
# ─────────────────────────────────────────────────────────────────────────────

def normvecs(pts):
    n=len(pts); tang=np.zeros((n,2))
    for i in range(n):
        d=pts[(i+1)%n]-pts[(i-1)%n]; l=np.linalg.norm(d)
        tang[i]=d/l if l>1e-9 else np.array([1.,0.])
    return np.column_stack([-tang[:,1],tang[:,0]])


# ─────────────────────────────────────────────────────────────────────────────
# 7. FRICTION-CIRCLE VELOCITY PROFILE
#    This is the key upgrade over simple ay_max limiting.
#    Uses the combined acceleration circle: (ax/ax_lim)² + (ay/ay_max)² ≤ 1
#    Corner speed: v = sqrt(ay_max * R)  where R = 1/|kappa|
#    But ay_available reduces as ax is used → friction circle coupling
# ─────────────────────────────────────────────────────────────────────────────

def friction_circle_velocity_profile(kappa, ds, veh: Vehicle, n_passes=3):
    """
    Forward-backward velocity integration with friction circle constraint.
    Multiple passes to converge the coupled ax-ay problem.
    """
    n   = len(kappa)
    eps = 1e-6

    # Max cornering speed from lateral limit only (upper bound)
    v_corner = np.where(
        np.abs(kappa) > eps,
        np.sqrt(veh.ay_max / (np.abs(kappa) + eps)),
        veh.v_max)
    v_corner = np.clip(v_corner, veh.v_min, veh.v_max)

    vx = v_corner.copy()

    for _ in range(n_passes):
        # Forward pass — acceleration limited with friction circle
        vf = np.zeros(n); vf[0] = veh.v_min
        for i in range(1, n):
            d    = ds[(i-1)%n]
            # ay used at current speed and curvature
            ay_used = vf[i-1]**2 * abs(kappa[(i-1)%n])
            # available ax from friction circle
            ay_ratio = min(ay_used / (veh.ay_max + eps), 1.0)
            ax_avail = veh.ax_max * math.sqrt(max(1.0 - ay_ratio**2, 0.0))
            v_possible = math.sqrt(max(vf[i-1]**2 + 2*ax_avail*d, 0.0))
            vf[i] = min(v_possible, v_corner[i], veh.v_max)

        # Backward pass — braking limited with friction circle
        vb = np.zeros(n); vb[-1] = vf[-1]
        for i in range(n-2, -1, -1):
            d    = ds[i]
            ay_used = vb[i+1]**2 * abs(kappa[(i+1)%n])
            ay_ratio = min(ay_used / (veh.ay_max + eps), 1.0)
            ax_avail = abs(veh.ax_min) * math.sqrt(max(1.0 - ay_ratio**2, 0.0))
            v_possible = math.sqrt(max(vb[i+1]**2 + 2*ax_avail*d, 0.0))
            vb[i] = min(v_possible, v_corner[i], veh.v_max)

        vx = np.clip(np.minimum(vf, vb), veh.v_min, veh.v_max)

        # Update corner speed with achieved vx (iterative refinement)
        v_corner = np.minimum(v_corner, vx)

    # Acceleration profile
    ax = np.array([
        (vx[(i+1)%n]**2 - vx[i]**2) / (2*ds[i]+eps)
        for i in range(n)])
    ax = np.clip(ax, veh.ax_min, veh.ax_max)

    return vx, ax


# ─────────────────────────────────────────────────────────────────────────────
# 8. SPEED-WEIGHTED MINIMUM LAP TIME QP
#    Key difference from min-curvature:
#    The Hessian H is weighted by local speed squared → high-speed zones
#    pay a much larger penalty for curvature, so the optimiser preferentially
#    widens the line where the car is fastest (where it matters most for time).
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlt_qp(refline, nv, wr, wl, vx):
    """Speed-weighted minimum-lap-time QP matrices."""
    n  = len(refline)
    dx = np.diff(np.append(refline[:,0], refline[0,0]))
    dy = np.diff(np.append(refline[:,1], refline[1,1]))
    ds = np.mean(np.sqrt(dx**2+dy**2))

    D = np.zeros((n,n))
    for i in range(n):
        D[i,(i-1)%n]=1.; D[i,i]=-2.; D[i,(i+1)%n]=1.
    D /= ds**2

    # Speed-squared weighting: penalise curvature more at high speed
    # W = diag(vx²) — faster sections get higher weight
    W = np.diag(vx**2 / (vx.max()**2 + 1e-9))   # normalised 0..1

    Nx = np.diag(nv[:,0]); Ny = np.diag(nv[:,1])
    Ax = D@Nx; Ay = D@Ny

    # Weighted Hessian
    H = 2.*(Ax.T@W@Ax + Ay.T@W@Ay)
    rx = refline[:,0]; ry = refline[:,1]
    f  = 2.*(Ax.T@W@(D@rx) + Ay.T@W@(D@ry))
    return H, f, -wr, wl


def _solve(H, f, lb, ub):
    n = len(f)
    if _QP == "quadprog":
        G = H + np.eye(n)*1e-8
        C = np.vstack([np.eye(n),-np.eye(n)]).T
        b = np.concatenate([lb,-ub])
        try:
            sol,*_ = quadprog.solve_qp(G,-f,C,b); return sol
        except Exception as e:
            warnings.warn(f"quadprog: {e}"); return np.zeros(n)
    elif _QP == "osqp":
        import osqp, scipy.sparse as sps
        prob = osqp.OSQP()
        prob.setup(sps.csc_matrix(H+np.eye(n)*1e-8), f,
                   sps.eye(n,"csc"), lb, ub,
                   verbose=False, eps_abs=1e-6, max_iter=20000)
        r = prob.solve()
        return r.x if r.info.status=="solved" else np.zeros(n)
    return np.zeros(n)


def min_laptime_optimise(refline, nv, wr, wl, veh: Vehicle,
                          n_outer=5, n_qp=3):
    """
    Outer loop: alternates between
      (a) speed-weighted QP → optimise raceline given current vx
      (b) friction-circle velocity profile → update vx given new raceline
    This joint optimisation converges to the minimum lap time line.
    """
    alpha = np.zeros(len(refline))

    # Initial velocity profile on centreline
    s, psi, kappa, ds = _geometry(refline)
    vx, ax = friction_circle_velocity_profile(kappa, ds, veh)

    for outer in range(n_outer):
        rl = refline + alpha[:,None]*nv

        # Inner QP iterations with current speed weighting
        for inner in range(n_qp):
            H, f, lb, ub = _build_mlt_qp(rl, nv, wr, wl, vx)
            delta  = _solve(H, f, lb, ub)
            alpha  = np.clip(alpha+delta, -wr, wl)
            rl     = refline + alpha[:,None]*nv
            change = np.max(np.abs(delta))
            if change < 1e-5: break

        # Recompute velocity profile on new raceline
        s, psi, kappa, ds = _geometry(rl)
        vx_new, ax_new = friction_circle_velocity_profile(kappa, ds, veh)
        lap = float(np.sum(ds/(vx_new+1e-9)))

        print(f"  [outer {outer+1}/{n_outer}] "
              f"max_Δα={change:.4f}m  "
              f"lap={lap:.3f}s  "
              f"v={vx_new.min():.2f}–{vx_new.max():.2f} m/s")
        vx = vx_new

    return alpha, rl, vx, ax_new


# ─────────────────────────────────────────────────────────────────────────────
# 9. GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def _geometry(rl):
    n=len(rl)
    dx=np.diff(np.append(rl[:,0],rl[0,0]))
    dy=np.diff(np.append(rl[:,1],rl[0,1]))
    ds=np.sqrt(dx**2+dy**2)
    psi=(np.arctan2(dy,dx)+np.pi)%(2*np.pi)-np.pi
    kappa=np.zeros(n)
    for i in range(n):
        p1=rl[(i-1)%n];p2=rl[i];p3=rl[(i+1)%n]
        d1=p2-p1;d2=p3-p2
        cross=d1[0]*d2[1]-d1[1]*d2[0]
        denom=np.linalg.norm(d1)*np.linalg.norm(d2)*np.linalg.norm(p3-p1)
        kappa[i]=2.*abs(cross)/(denom+1e-12)*np.sign(cross)
    s=np.concatenate([[0],np.cumsum(ds[:-1])])
    return s,psi,kappa,ds


# ─────────────────────────────────────────────────────────────────────────────
# 10. OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path, rl, wr, wl, psi, kappa, vx, ax):
    np.savetxt(path,
               np.column_stack([rl,wr,wl,psi,kappa,vx,ax]),
               delimiter=",",
               header="x_m,y_m,w_tr_right_m,w_tr_left_m,psi_rad,kappa_radpm,vx_mps,ax_mps2",
               comments="", fmt="%.6f")
    print(f"[out] CSV: {path}  ({len(rl)} pts)")


def visualise(md, refline_w, raceline_w, vx, path):
    res=md["resolution"]; ox=md["origin_x"]; oy=md["origin_y"]
    H=md["H"]; W=md["W"]; grid=md["grid"]
    ext=[ox,ox+W*res,oy,oy+H*res]
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    for ai,(ax_obj,title) in enumerate(zip(axes,["Map + Raceline","Velocity Profile"])):
        ax_obj.imshow(np.flipud(grid),cmap="gray",origin="lower",extent=ext)
        ax_obj.set_aspect("equal"); ax_obj.set_xlabel("x[m]"); ax_obj.set_ylabel("y[m]")
        ax_obj.set_title(title)
        if ai==0:
            ax_obj.plot(refline_w[:,0],refline_w[:,1],"c--",lw=1,alpha=0.5,label="Centreline")
            ax_obj.plot(raceline_w[:,0],raceline_w[:,1],"r-",lw=2.5,label="Min-LapTime Raceline")
            ax_obj.legend(fontsize=8)
        else:
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
            seg=np.array([raceline_w[:-1],raceline_w[1:]]).transpose(1,0,2)
            seg=np.vstack([seg,[[raceline_w[-1],raceline_w[0]]]])
            norm=Normalize(vmin=vx.min(),vmax=vx.max())
            lc=LineCollection(seg,cmap="RdYlGn",norm=norm,linewidth=2.5)
            lc.set_array(vx); ax_obj.add_collection(lc)
            plt.colorbar(lc,ax=ax_obj,label="Speed [m/s]",fraction=0.04)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"[out] PNG: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(pgm, yaml_path=None, out="output", step=0.05, safety=0.05,
        n_outer=5, n_qp=3, veh=None):
    if veh is None: veh = Vehicle()
    os.makedirs(out, exist_ok=True)

    print("\n[1/7] Loading map...")
    md  = load_map(pgm, yaml_path)
    res = md["resolution"]; ox=md["origin_x"]; oy=md["origin_y"]; H=md["H"]
    print(f"      {md['W']}x{H} @ {res}m/px  origin=({ox:.3f},{oy:.3f})")
    print(f"      World: x=[{ox:.2f},{ox+md['W']*res:.2f}]  y=[{oy:.2f},{oy+H*res:.2f}]")

    print("[2/7] Track mask...")
    mask = extract_track_mask(md, safety)
    cv2.imwrite(os.path.join(out,"debug_mask.png"), mask)

    print("[3/7] Skeleton...")
    cl_px, _ = extract_centreline_px(mask)
    cl_w = pts_px_to_world(cl_px, res, ox, oy, H)
    print(f"      {len(cl_w)} skeleton pts")

    print("[4/7] Smooth & resample...")
    refline, L = smooth_resample(cl_w, step)
    print(f"      {len(refline)} pts  total={L:.2f}m")
    print(f"      Theoretical minimum @ v_max={veh.v_max:.1f}m/s: {L/veh.v_max:.2f}s")

    ref_px = pts_world_to_px(refline, res, ox, oy, H)

    print("[5/7] Track widths...")
    wr, wl = track_widths(ref_px, mask, res, safety)
    print(f"      w_right={wr.min():.2f}–{wr.max():.2f}  "
          f"w_left={wl.min():.2f}–{wl.max():.2f} m")

    nv = normvecs(refline)

    print(f"[6/7] Min-lap-time optimisation "
          f"(outer={n_outer}, qp={n_qp}, backend={_QP})...")
    print(f"      Vehicle: v_max={veh.v_max} ay_max={veh.ay_max} "
          f"ax=[{veh.ax_min},{veh.ax_max}] mu={veh.mu}")
    alpha, raceline, vx, ax = min_laptime_optimise(
        refline, nv, wr, wl, veh, n_outer, n_qp)

    wr_rl = wr - alpha; wl_rl = wl + alpha

    print("[7/7] Final geometry...")
    s, psi, kappa, ds = _geometry(raceline)
    lap = float(np.sum(ds/(vx+1e-9)))
    dist = float(np.sum(ds))
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Track length   : {dist:.2f} m            │")
    print(f"  │  Theoretical lap: {lap:.3f} s           │")
    print(f"  │  Avg speed      : {dist/lap:.2f} m/s        │")
    print(f"  │  Max speed      : {vx.max():.2f} m/s        │")
    print(f"  │  Min speed      : {vx.min():.2f} m/s        │")
    print(f"  └─────────────────────────────────────┘")

    # Apply origin
    rl_w = raceline.copy()
    rf_w = refline.copy()

    write_csv(os.path.join(out,"raceline.csv"),
              rl_w, wr_rl, wl_rl, psi, kappa, vx, ax)
    visualise(md, rf_w, rl_w, vx, os.path.join(out,"raceline.png"))
    return rl_w, lap


def main():
    p = argparse.ArgumentParser(
        description="Minimum Lap Time Raceline Generator — AutoDRIVE RoboRacer 2026")
    p.add_argument("--pgm",     required=True)
    p.add_argument("--yaml",    default=None)
    p.add_argument("--output",  default="output")
    p.add_argument("--step",    type=float, default=0.05,
                   help="Waypoint spacing [m] (smaller=smoother, default 0.05)")
    p.add_argument("--safety",  type=float, default=0.05,
                   help="Wall safety margin [m] (default 0.05)")
    p.add_argument("--n_outer", type=int,   default=5,
                   help="Outer iterations (raceline↔velocity, default 5)")
    p.add_argument("--n_qp",    type=int,   default=3,
                   help="QP iterations per outer loop (default 3)")
    p.add_argument("--v_max",   type=float, default=22.88)
    p.add_argument("--ax_max",  type=float, default=8.0)
    p.add_argument("--ax_min",  type=float, default=-10.0)
    p.add_argument("--ay_max",  type=float, default=9.8)
    p.add_argument("--mu",      type=float, default=1.0)
    args = p.parse_args()

    veh = Vehicle()
    veh.v_max  = args.v_max
    veh.ax_max = args.ax_max
    veh.ax_min = args.ax_min
    veh.ay_max = args.ay_max
    veh.mu     = args.mu
    veh.a_total_max = args.mu * 9.81

    run(args.pgm, args.yaml, args.output,
        args.step, args.safety, args.n_outer, args.n_qp, veh)


if __name__ == "__main__":
    main()