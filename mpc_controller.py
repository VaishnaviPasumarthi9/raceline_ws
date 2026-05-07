#!/usr/bin/env python3
"""
Global Raceline Generator — Direct Collocation (IPOPT)
======================================================
Optimizes the entire lap simultaneously to minimize T = integral(1/v) ds.
Uses a spatial bicycle model formulation.
"""

import numpy as np
import casadi as ca
import pandas as pd
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Vehicle & Track Parameters
# ─────────────────────────────────────────────────────────────────────────────
WB        = 0.3240    # [m]
V_MAX     = 22.88     # [m/s]
STEER_MAX = 0.5236    # [rad]
AY_MAX    = 9.8       # [m/s²] (Lateral Grip)
AX_MAX    = 8.0       # [m/s²] (Longitudinal Accel)
AX_MIN    = -10.0     # [m/s²] (Braking)
TRACK_W   = 3.0       # [m] Total width (1.5m left/right)

def generate_raceline(csv_path, output_path):
    # 1. Load Centerline
    data = pd.read_csv(csv_path)
    # Assume CSV has x, y coordinates
    cx = data['x'].values
    cy = data['y'].values
    
    N = len(cx)
    # Calculate track distance and curvature
    dx = np.gradient(cx)
    dy = np.gradient(cy)
    ds = np.sqrt(dx**2 + dy**2)
    s_cum = np.cumsum(ds)
    L = s_cum[-1]
    
    # Calculate Curvature (kappa)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(1.5)

    # 2. Setup Optimization Problem
    opti = ca.Opti()

    # State variables (Spatial Domain):
    # n: lateral deviation from centerline
    # alpha: heading error relative to centerline
    # v: velocity
    # t: time
    n = opti.variable(N)
    alpha = opti.variable(N)
    v = opti.variable(N)
    t = opti.variable(N)

    # Control variables:
    # delta: steering angle
    # a: longitudinal acceleration
    delta = opti.variable(N)
    a = opti.variable(N)

    # 3. Objective: Minimize Total Lap Time
    opti.minimize(t[N-1])

    # 4. Direct Collocation Constraints (Trapezoidal Integration)
    # Using spatial dynamics: dx/ds = f(x, u) / (s_dot)
    # where s_dot = (v * cos(alpha)) / (1 - n*kappa)
    
    for k in range(N-1):
        # Local curvature
        k_ref = kappa[k]
        dist = ds[k]
        
        # Scaling factor (Relates centerline progress to vehicle progress)
        sf = (1 - n[k] * k_ref) / (v[k] * ca.cos(alpha[k]) + 1e-6)

        # Dynamics (f = dx/ds)
        dn_ds = ca.tan(alpha[k]) * (1 - n[k] * k_ref)
        da_ds = ( (ca.tan(delta[k]) / WB) - (k_ref / (1 - n[k] * k_ref)) ) * sf
        dv_ds = a[k] * sf
        dt_ds = sf

        # Trapezoidal Collocation
        opti.subject_to(n[k+1] == n[k] + dist * dn_ds)
        opti.subject_to(alpha[k+1] == alpha[k] + dist * da_ds)
        opti.subject_to(v[k+1] == v[k] + dist * dv_ds)
        opti.subject_to(t[k+1] == t[k] + dist * dt_ds)

    # 5. Boundary & Path Constraints
    # Track Limits (stay within width)
    opti.subject_to(opti.bounded(-TRACK_W/2 + 0.2, n, TRACK_W/2 - 0.2))
    
    # Vehicle Limits
    opti.subject_to(opti.bounded(0.5, v, V_MAX))
    opti.subject_to(opti.bounded(-STEER_MAX, delta, STEER_MAX))
    opti.subject_to(opti.bounded(AX_MIN, a, AX_MAX))

    # Friction Circle Constraint (Combined Grip)
    # (a_long / a_max)^2 + (a_lat / a_max)^2 <= 1
    a_lat = (v**2 * ca.tan(delta)) / WB
    opti.subject_to((a/AX_MAX)**2 + (a_lat/AY_MAX)**2 <= 1)

    # Continuity (Closed loop)
    opti.subject_to(n[0] == n[N-1])
    opti.subject_to(alpha[0] == alpha[N-1])
    opti.subject_to(v[0] == v[N-1])
    opti.subject_to(t[0] == 0)

    # 6. Solver (IPOPT)
    opts = {
        "ipopt.max_iter": 2000,
        "ipopt.print_level": 5,
        "ipopt.tol": 1e-4,
    }
    opti.solver("ipopt", opts)

    # Initial Guess
    opti.set_initial(v, V_MAX * 0.5)
    opti.set_initial(n, 0)

    sol = opti.solve()

    # 7. Post-process and Save
    # Transform Frenet (n) back to Global (x, y)
    n_opt = sol.value(n)
    v_opt = sol.value(v)
    
    # Calculate unit normals of centerline for reconstruction
    nx = -dy / ds
    ny = dx / ds
    
    rx = cx + n_opt * nx
    ry = cy + n_opt * ny
    
    # Export to CSV for your MPC controller
    df_out = pd.DataFrame({
        'x': rx,
        'y': ry,
        'vx_ref': v_opt,
        'psi_ref': sol.value(alpha) # This is heading error; you may need to add track heading
    })
    df_out.to_csv(output_path, index=False)
    print(f"Optimal raceline saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input centerline CSV")
    parser.add_argument("--out", default="optimized_raceline.csv")
    args = parser.parse_args()
    generate_raceline(args.csv, args.out)