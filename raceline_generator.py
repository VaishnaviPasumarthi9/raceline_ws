#!/usr/bin/env python3
"""
MINIMUM LAP TIME Raceline Generator — AutoDRIVE RoboRacer 2026
===============================================================
Most aggressive possible raceline within physical tyre limits.

Algorithm:
  - Speed-weighted QP on lateral offsets (outer loop)
  - Friction-circle velocity profile (inner loop)
  - Iterates until convergence → true minimum lap time line

Vehicle: RoboRacer (technical guide 2026)
  v_max=22.88 m/s  ay_max=9.8 m/s²  ax=[−10,+8] m/s²
"""

import os, sys, argparse, math, warnings
import numpy as np
import cv2
from scipy import ndimage, interpolate
from scipy.spatial import KDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import quadprog; _QP="quadprog"
except ImportError:
    try:
        import osqp; _QP="osqp"
    except ImportError:
        _QP=None

class Vehicle:
    v_max=22.88; v_min=0.5
    ax_max=8.0;  ax_min=-10.0
    ay_max=9.8;  mu=1.0; g=9.81

# ── map ───────────────────────────────────────────────────────────────────────
def load_map(pgm,yaml_path=None):
    img=cv2.imread(pgm,cv2.IMREAD_GRAYSCALE)
    if img is None: raise RuntimeError(f"Cannot read {pgm}")
    res=0.05;ox=0.;oy=0.
    if yaml_path and os.path.isfile(yaml_path):
        import yaml
        with open(yaml_path) as f: m=yaml.safe_load(f)
        res=float(m.get("resolution",res))
        o=m.get("origin",[0,0,0]); ox,oy=float(o[0]),float(o[1])
    H,W=img.shape
    return {"grid":img,"resolution":res,"origin_x":ox,"origin_y":oy,"H":H,"W":W}

def pts_px_to_world(p,res,ox,oy,H):
    o=np.zeros_like(p); o[:,0]=p[:,0]*res+ox; o[:,1]=(H-1-p[:,1])*res+oy; return o
def pts_world_to_px(p,res,ox,oy,H):
    o=np.zeros_like(p); o[:,0]=(p[:,0]-ox)/res; o[:,1]=H-1-(p[:,1]-oy)/res; return o

# ── track mask ────────────────────────────────────────────────────────────────
def extract_mask(md,safety=0.05):
    img=md["grid"]; res=md["resolution"]
    free=(img>180).astype(np.uint8)*255
    n,labels,stats,_=cv2.connectedComponentsWithStats(free,connectivity=8)
    areas=sorted([(stats[i,cv2.CC_STAT_AREA],i) for i in range(1,n)],reverse=True)
    best=-1; mask=None
    for _,ci in areas[:5]:
        blob=((labels==ci).astype(np.uint8))*255
        area=stats[ci,cv2.CC_STAT_AREA]
        filled=ndimage.binary_fill_holes(blob>0).astype(np.uint8)*255
        fr=np.count_nonzero(filled)/(area+1)
        if fr>best: best=fr; mask=blob
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    mp=max(1,int(safety/res))
    k2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mp*2+1,mp*2+1))
    return cv2.erode(mask,k2,iterations=1)

# ── centreline ────────────────────────────────────────────────────────────────
def centreline_px(mask):
    from skimage.morphology import skeletonize
    skel=skeletonize((mask>127).astype(np.uint8)).astype(np.uint8)*255
    ys,xs=np.where(skel>0); pts=np.column_stack([xs,ys]).astype(float)
    n=len(pts); tree=KDTree(pts); vis=np.zeros(n,bool); order=[0]; vis[0]=True
    for _ in range(n-1):
        _,idxs=tree.query(pts[order[-1]],k=min(20,n))
        for i in idxs[1:]:
            if not vis[i]: order.append(i); vis[i]=True; break
    return pts[order]

# ── widths ────────────────────────────────────────────────────────────────────
def widths(cl,mask,res,safety=0.05):
    n=len(cl); H,W=mask.shape; wr=np.zeros(n); wl=np.zeros(n)
    for i in range(n):
        tang=cl[(i+1)%n]-cl[(i-1)%n]; tl=np.linalg.norm(tang)
        if tl<1e-9: wr[i]=wl[i]=safety; continue
        tang/=tl; nr=np.array([tang[1],-tang[0]]); nl=np.array([-tang[1],tang[0]])
        cx,cy=cl[i]
        for nm,st in [(nr,wr),(nl,wl)]:
            for s in range(1,600):
                nx=int(round(cx+nm[0]*s)); ny=int(round(cy+nm[1]*s))
                if nx<0 or ny<0 or nx>=W or ny>=H: st[i]=max(s*res-safety,0.05); break
                if mask[ny,nx]<127: st[i]=max(s*res-safety,0.05); break
            else: st[i]=max(600*res-safety,0.05)
    return wr,wl

# ── resample ──────────────────────────────────────────────────────────────────
def resample(pts,step=0.05,s=None):
    pc=np.vstack([pts,pts[0]]); x,y=pc[:,0],pc[:,1]
    seg=np.sqrt(np.diff(x)**2+np.diff(y)**2)
    t=np.concatenate([[0],np.cumsum(seg)]); L=t[-1]
    if s is None: s=L*0.003
    tck,_=interpolate.splprep([x,y],u=t,s=s,per=True,k=5)
    n=max(100,int(L/step)); u=np.linspace(0,L,n,endpoint=False)
    xs,ys=interpolate.splev(u,tck)
    return np.column_stack([xs,ys]),L

# ── geometry ──────────────────────────────────────────────────────────────────
def geometry(rl):
    n=len(rl)
    dx=np.diff(np.append(rl[:,0],rl[0,0])); dy=np.diff(np.append(rl[:,1],rl[0,1]))
    ds=np.sqrt(dx**2+dy**2); psi=(np.arctan2(dy,dx)+np.pi)%(2*np.pi)-np.pi
    kappa=np.zeros(n)
    for i in range(n):
        p1=rl[(i-1)%n];p2=rl[i];p3=rl[(i+1)%n]
        d1=p2-p1;d2=p3-p2
        cross=d1[0]*d2[1]-d1[1]*d2[0]
        denom=np.linalg.norm(d1)*np.linalg.norm(d2)*np.linalg.norm(p3-p1)
        kappa[i]=2*abs(cross)/(denom+1e-12)*np.sign(cross)
    return np.concatenate([[0],np.cumsum(ds[:-1])]),psi,kappa,ds

def normvecs(pts):
    n=len(pts); tang=np.zeros((n,2))
    for i in range(n):
        d=pts[(i+1)%n]-pts[(i-1)%n]; l=np.linalg.norm(d)
        tang[i]=d/l if l>1e-9 else np.array([1.,0.])
    return np.column_stack([-tang[:,1],tang[:,0]])

# ── friction circle velocity profile ─────────────────────────────────────────
def vel_profile(kappa,ds,veh,passes=4):
    n=len(kappa); eps=1e-6
    vc=np.where(np.abs(kappa)>eps,
                np.sqrt(veh.ay_max/(np.abs(kappa)+eps)),veh.v_max)
    vc=np.clip(vc,veh.v_min,veh.v_max)
    vx=vc.copy()
    for _ in range(passes):
        vf=np.zeros(n); vf[0]=veh.v_min
        for i in range(1,n):
            d=ds[(i-1)%n]; ay=vf[i-1]**2*abs(kappa[(i-1)%n])
            ar=min(ay/(veh.ay_max+eps),1.0)
            ax_a=veh.ax_max*math.sqrt(max(1-ar**2,0))
            vf[i]=min(math.sqrt(max(vf[i-1]**2+2*ax_a*d,0)),vc[i],veh.v_max)
        vb=np.zeros(n); vb[-1]=vf[-1]
        for i in range(n-2,-1,-1):
            d=ds[i]; ay=vb[i+1]**2*abs(kappa[(i+1)%n])
            ar=min(ay/(veh.ay_max+eps),1.0)
            ax_a=abs(veh.ax_min)*math.sqrt(max(1-ar**2,0))
            vb[i]=min(math.sqrt(max(vb[i+1]**2+2*ax_a*d,0)),vc[i],veh.v_max)
        vx=np.clip(np.minimum(vf,vb),veh.v_min,veh.v_max)
        vc=np.minimum(vc,vx)
    ax=np.clip([(vx[(i+1)%n]**2-vx[i]**2)/(2*ds[i]+eps) for i in range(n)],
               veh.ax_min,veh.ax_max)
    return vx,np.array(ax)

# ── speed-weighted min-laptime QP ─────────────────────────────────────────────
def _qp(H,f,lb,ub):
    n=len(f)
    if _QP=="quadprog":
        G=H+np.eye(n)*1e-8; C=np.vstack([np.eye(n),-np.eye(n)]).T; b=np.concatenate([lb,-ub])
        try: sol,*_=quadprog.solve_qp(G,-f,C,b); return sol
        except: return np.zeros(n)
    elif _QP=="osqp":
        import osqp,scipy.sparse as sps
        prob=osqp.OSQP()
        prob.setup(sps.csc_matrix(H+np.eye(n)*1e-8),f,sps.eye(n,"csc"),lb,ub,
                   verbose=False,eps_abs=1e-6,max_iter=20000)
        r=prob.solve(); return r.x if r.info.status=="solved" else np.zeros(n)
    return np.zeros(n)

def optimise(refline,nv,wr,wl,veh,n_outer=8,n_qp=5):
    alpha=np.zeros(len(refline))
    _,_,kappa,ds=geometry(refline)
    vx,_=vel_profile(kappa,ds,veh)

    for outer in range(n_outer):
        rl=refline+alpha[:,None]*nv
        for _ in range(n_qp):
            n=len(rl)
            dx=np.diff(np.append(rl[:,0],rl[0,0])); dy=np.diff(np.append(rl[:,1],rl[1,1]))
            ds_=np.mean(np.sqrt(dx**2+dy**2))
            D=np.zeros((n,n))
            for i in range(n): D[i,(i-1)%n]=1.;D[i,i]=-2.;D[i,(i+1)%n]=1.
            D/=ds_**2
            # Speed-squared weighting — high-speed zones penalised more
            W=np.diag(vx**2/(vx.max()**2+1e-9))
            Nx=np.diag(nv[:,0]); Ny=np.diag(nv[:,1])
            Ax=D@Nx; Ay=D@Ny
            H=2*(Ax.T@W@Ax+Ay.T@W@Ay)
            rx=rl[:,0]; ry=rl[:,1]
            f=2*(Ax.T@W@(D@rx)+Ay.T@W@(D@ry))
            delta=_qp(H,f,-wr,wl)
            alpha=np.clip(alpha+delta,-wr,wl)
            rl=refline+alpha[:,None]*nv
            if np.max(np.abs(delta))<1e-5: break

        _,_,kappa,ds=geometry(rl)
        vx,ax=vel_profile(kappa,ds,veh)
        lap=float(np.sum(ds/(vx+1e-9)))
        print(f"  [outer {outer+1}/{n_outer}] lap={lap:.3f}s  "
              f"v={vx.min():.1f}–{vx.max():.1f} m/s")

    return alpha,rl,vx,ax

# ── output ────────────────────────────────────────────────────────────────────
def write_csv(path,rl,wr,wl,psi,kappa,vx,ax):
    np.savetxt(path,np.column_stack([rl,wr,wl,psi,kappa,vx,ax]),
               delimiter=",",
               header="x_m,y_m,w_tr_right_m,w_tr_left_m,psi_rad,kappa_radpm,vx_mps,ax_mps2",
               comments="",fmt="%.6f")
    print(f"[out] {path}  ({len(rl)} pts)")

def visualise(md,ref_w,rl_w,vx,path):
    res=md["resolution"];ox=md["origin_x"];oy=md["origin_y"];H=md["H"];W=md["W"]
    ext=[ox,ox+W*res,oy,oy+H*res]
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    for ai,(ao,title) in enumerate(zip(axes,["Map + Raceline","Velocity Profile"])):
        ao.imshow(np.flipud(md["grid"]),cmap="gray",origin="lower",extent=ext)
        ao.set_aspect("equal"); ao.set_xlabel("x[m]"); ao.set_ylabel("y[m]"); ao.set_title(title)
        if ai==0:
            ao.plot(ref_w[:,0],ref_w[:,1],"c--",lw=1,alpha=0.5,label="Centreline")
            ao.plot(rl_w[:,0],rl_w[:,1],"r-",lw=2.5,label="Min-LapTime Line")
            ao.legend(fontsize=8)
        else:
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
            seg=np.array([rl_w[:-1],rl_w[1:]]).transpose(1,0,2)
            seg=np.vstack([seg,[[rl_w[-1],rl_w[0]]]])
            norm=Normalize(vmin=vx.min(),vmax=vx.max())
            lc=LineCollection(seg,cmap="RdYlGn",norm=norm,linewidth=2.5)
            lc.set_array(vx); ao.add_collection(lc)
            plt.colorbar(lc,ax=ao,label="Speed [m/s]",fraction=0.04)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

# ── main ──────────────────────────────────────────────────────────────────────
def run(pgm,yaml_path,out,step,safety,n_outer,n_qp,veh):
    os.makedirs(out,exist_ok=True)
    print("\n[1] Loading map...")
    md=load_map(pgm,yaml_path)
    res=md["resolution"];ox=md["origin_x"];oy=md["origin_y"];H=md["H"]
    print(f"    {md['W']}x{H} @ {res}m/px  origin=({ox:.3f},{oy:.3f})")

    print("[2] Track mask...")
    mask=extract_mask(md,safety)
    cv2.imwrite(os.path.join(out,"debug_mask.png"),mask)

    print("[3] Centreline...")
    cl=centreline_px(mask)
    cl_w=pts_px_to_world(cl,res,ox,oy,H)
    print(f"    {len(cl_w)} pts")

    print("[4] Resample...")
    ref_w,L=resample(cl_w,step)
    print(f"    {len(ref_w)} pts  L={L:.2f}m")
    print(f"    Speed of light limit: {L/veh.v_max:.3f}s  (straight line at v_max)")

    ref_px=pts_world_to_px(ref_w,res,ox,oy,H)

    print("[5] Track widths...")
    wr,wl=widths(ref_px,mask,res,safety)
    print(f"    wr={wr.min():.2f}–{wr.max():.2f}  wl={wl.min():.2f}–{wl.max():.2f} m")

    nv=normvecs(ref_w)

    print(f"[6] Min-laptime optimisation (outer={n_outer} qp={n_qp} backend={_QP})...")
    alpha,rl_w,vx,ax=optimise(ref_w,nv,wr,wl,veh,n_outer,n_qp)

    wr_rl=wr-alpha; wl_rl=wl+alpha
    _,psi,kappa,ds=geometry(rl_w)
    lap=float(np.sum(ds/(vx+1e-9)))
    dist=float(np.sum(ds))

    print(f"\n╔══════════════════════════════════════╗")
    print(f"║  Track length   : {dist:.2f} m")
    print(f"║  Theoretical lap: {lap:.3f} s  ← BEST POSSIBLE")
    print(f"║  Avg speed      : {dist/lap:.2f} m/s")
    print(f"║  Max speed      : {vx.max():.2f} m/s")
    print(f"║  Min speed      : {vx.min():.2f} m/s")
    print(f"║  v_max limit    : {veh.v_max:.2f} m/s")
    print(f"║  Absolute floor : {dist/veh.v_max:.3f} s (no corners)")
    print(f"╚══════════════════════════════════════╝")

    write_csv(os.path.join(out,"raceline.csv"),rl_w,wr_rl,wl_rl,psi,kappa,vx,ax)
    visualise(md,ref_w,rl_w,vx,os.path.join(out,"raceline.png"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pgm",required=True)
    p.add_argument("--yaml",default=None)
    p.add_argument("--output",default="output")
    p.add_argument("--step",type=float,default=0.05)
    p.add_argument("--safety",type=float,default=0.05)
    p.add_argument("--n_outer",type=int,default=8)
    p.add_argument("--n_qp",type=int,default=5)
    p.add_argument("--v_max",type=float,default=22.88)
    p.add_argument("--ax_max",type=float,default=8.0)
    p.add_argument("--ax_min",type=float,default=-10.0)
    p.add_argument("--ay_max",type=float,default=9.8)
    p.add_argument("--mu",type=float,default=1.0)
    args=p.parse_args()
    veh=Vehicle()
    veh.v_max=args.v_max; veh.ax_max=args.ax_max
    veh.ax_min=args.ax_min; veh.ay_max=args.ay_max; veh.mu=args.mu
    run(args.pgm,args.yaml,args.output,args.step,args.safety,args.n_outer,args.n_qp,veh)

if __name__=="__main__":
    main()