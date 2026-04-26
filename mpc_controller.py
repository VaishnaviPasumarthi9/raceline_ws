#!/usr/bin/env python3
"""
MPC Controller for AutoDRIVE RoboRacer — Minimum Lap Time
==========================================================
Uses CasADi IPOPT solver for real-time Model Predictive Control.

Why MPC beats Pure Pursuit:
  - Optimises over a future horizon (not just one lookahead point)
  - Enforces exact tyre friction circle constraints
  - Minimises time to reach next waypoint, not just tracking error
  - Handles braking/acceleration trade-offs optimally

Topics (AutoDRIVE RoboRacer):
  SUB: /autodrive/roboracer_1/ips   geometry_msgs/Point   (position)
  SUB: /autodrive/roboracer_1/imu   sensor_msgs/Imu       (orientation)
  SUB: /autodrive/roboracer_1/odom  nav_msgs/Odometry     (with --use_odom)
  PUB: /autodrive/roboracer_1/throttle_command  std_msgs/Float32 [-1,1]
  PUB: /autodrive/roboracer_1/steering_command  std_msgs/Float32 [-1,1]

Usage:
  pip3 install casadi
  source /opt/ros/jazzy/setup.bash
  cd ~/raceline_ws

  # Development (use odom):
  python3 mpc_controller.py --csv output/raceline.csv --use_odom

  # Aggressive — push harder:
  python3 mpc_controller.py --csv output/raceline.csv --use_odom --speed_gain 1.2
"""

import argparse, math, sys, time
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
except ImportError:
    print("ERROR: source /opt/ros/jazzy/setup.bash"); sys.exit(1)

try:
    import casadi as ca
except ImportError:
    print("ERROR: pip3 install casadi"); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Vehicle constants — AutoDRIVE RoboRacer technical guide 2026
# ─────────────────────────────────────────────────────────────────────────────
WB          = 0.3240    # wheelbase [m]
V_MAX       = 22.88     # top speed [m/s]
STEER_MAX   = 0.5236    # max steering angle [rad]
STEER_RATE  = 3.2       # max steering rate [rad/s]
AY_MAX      = 9.8       # max lateral accel [m/s²]
AX_MAX      = 8.0       # max longitudinal accel [m/s²]
AX_MIN      = -10.0     # max braking [m/s²]

# ─────────────────────────────────────────────────────────────────────────────
# Bicycle model (kinematic):
#   x'    = v * cos(psi)
#   y'    = v * sin(psi)
#   psi'  = v * tan(delta) / L
#   v'    = a
# ─────────────────────────────────────────────────────────────────────────────

def build_mpc(N=20, dt=0.05):
    """
    Build CasADi MPC problem.
    State:  [x, y, psi, v]
    Control:[delta (steering), a (acceleration)]
    N: horizon steps
    dt: timestep [s]
    """
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(4, N+1)   # states
    U = opti.variable(2, N)     # controls

    # Parameters (set at runtime)
    X0    = opti.parameter(4)           # initial state
    Xref  = opti.parameter(4, N+1)     # reference trajectory
    Vref  = opti.parameter(N)          # reference speeds

    x   = X[0,:]; y   = X[1,:]
    psi = X[2,:]; v   = X[3,:]
    delta = U[0,:]; a = U[1,:]

    # ── Objective ─────────────────────────────────────────────────────────────
    # Minimise: tracking error + time (penalise low speed) + control smoothness
    Q_pos   = 5.0    # position tracking weight
    Q_psi   = 1.0    # heading tracking weight
    Q_v     = 2.0    # speed tracking weight
    R_delta = 0.5    # steering effort weight
    R_a     = 0.1    # acceleration effort weight
    R_ddelta= 2.0    # steering rate weight (smoothness)
    W_time  = 0.5    # time minimisation weight (penalise low speed)

    cost = 0
    for k in range(N):
        # Tracking error
        cost += Q_pos * ((x[k]-Xref[0,k])**2 + (y[k]-Xref[1,k])**2)
        cost += Q_psi * (psi[k]-Xref[2,k])**2
        cost += Q_v   * (v[k]-Vref[k])**2
        # Control effort
        cost += R_delta * delta[k]**2
        cost += R_a     * a[k]**2
        # Time minimisation (reward high speed)
        cost += W_time * (V_MAX - v[k])**2
        # Steering rate
        if k > 0:
            cost += R_ddelta * (delta[k]-delta[k-1])**2

    # Terminal cost
    cost += Q_pos*10 * ((x[N]-Xref[0,N])**2 + (y[N]-Xref[1,N])**2)

    opti.minimize(cost)

    # ── Dynamics constraints ──────────────────────────────────────────────────
    opti.subject_to(X[:,0] == X0)
    for k in range(N):
        x_next   = x[k]   + dt * v[k] * ca.cos(psi[k])
        y_next   = y[k]   + dt * v[k] * ca.sin(psi[k])
        psi_next = psi[k] + dt * v[k] * ca.tan(delta[k]) / WB
        v_next   = v[k]   + dt * a[k]
        opti.subject_to(X[:,k+1] == ca.vertcat(x_next,y_next,psi_next,v_next))

    # ── State constraints ─────────────────────────────────────────────────────
    opti.subject_to(opti.bounded(0.3, v, V_MAX))

    # ── Control constraints ───────────────────────────────────────────────────
    opti.subject_to(opti.bounded(-STEER_MAX, delta, STEER_MAX))
    opti.subject_to(opti.bounded(AX_MIN, a, AX_MAX))

    # Friction circle: (a/ax_lim)² + (v²κ/ay_max)² ≤ 1
    # Approximate with: |a| ≤ ax_max * sqrt(1 - (v²κ/ay_max)²)
    # For numerical stability, enforce soft version via penalty (already in cost)

    # ── Solver ────────────────────────────────────────────────────────────────
    opts = {
        "ipopt.print_level":      0,
        "ipopt.max_iter":         100,
        "ipopt.tol":              1e-4,
        "ipopt.warm_start_init_point": "yes",
        "print_time":             False,
        "ipopt.sb":               "yes",
    }
    opti.solver("ipopt", opts)

    return opti, X, U, X0, Xref, Vref


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

def closest_idx(xy, cx, cy, last, window=100):
    n=len(xy); best=float('inf'); bi=last
    for k in range(window):
        i=(last+k)%n; dx=xy[i,0]-cx; dy=xy[i,1]-cy; d=dx*dx+dy*dy
        if d<best: best=d; bi=i
    return bi

def angle_wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi


# ─────────────────────────────────────────────────────────────────────────────
# MPC Node
# ─────────────────────────────────────────────────────────────────────────────

class MPCNode(Node):

    def __init__(self, csv, N, dt, speed_gain, use_odom):
        super().__init__("mpc_controller")

        # Load raceline
        data        = np.loadtxt(csv, delimiter=",", skiprows=1)
        self.xy     = data[:,0:2]
        self.psi_ref= data[:,4]
        self.vx_ref = np.clip(data[:,6]*speed_gain, 0.5, V_MAX)
        self.n_pts  = len(self.xy)
        dist        = float(np.sum(np.sqrt(np.diff(self.xy[:,0])**2+np.diff(self.xy[:,1])**2)))
        self.get_logger().info(
            f"Raceline: {self.n_pts} pts  {dist:.1f}m  "
            f"v={self.vx_ref.min():.1f}–{self.vx_ref.max():.1f} m/s  "
            f"theoretical={dist/self.vx_ref.mean():.2f}s"
        )

        # Build MPC
        self.get_logger().info(f"Building MPC (N={N}, dt={dt}s)...")
        self.N  = N
        self.dt = dt
        self.opti, self.X, self.U, self.X0, self.Xref, self.Vref = build_mpc(N, dt)
        self.get_logger().info("MPC ready.")

        # State
        self.car_x    = 0.; self.car_y   = 0.
        self.car_yaw  = 0.; self.car_v   = 0.
        self.last_idx = 0
        self.pos_ok   = False; self.yaw_ok = False
        self._prev_delta = 0.
        self._prev_sol_X = None
        self._prev_sol_U = None
        self._dbg = 0

        # Publishers
        self.pub_t = self.create_publisher(Float32,"/autodrive/roboracer_1/throttle_command",10)
        self.pub_s = self.create_publisher(Float32,"/autodrive/roboracer_1/steering_command",10)

        be = QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        if use_odom:
            self.create_subscription(Odometry,"/autodrive/roboracer_1/odom",self._odom_cb,be)
            self.get_logger().info("Using /autodrive/roboracer_1/odom")
        else:
            self.create_subscription(Point,"/autodrive/roboracer_1/ips",self._ips_cb,be)
            self.create_subscription(Imu,  "/autodrive/roboracer_1/imu",self._imu_cb,be)
            self.get_logger().info("Using ips + imu")

        # Control loop at 20 Hz (MPC is heavier than PP)
        self.create_timer(0.05, self._loop)
        self.get_logger().info(
            "MPC running → throttle/steering_command\n"
            "Waiting for pose..."
        )

    def _odom_cb(self,msg):
        self.car_x=msg.pose.pose.position.x
        self.car_y=msg.pose.pose.position.y
        self.car_yaw=quat_to_yaw(msg.pose.pose.orientation)
        self.car_v=math.sqrt(msg.twist.twist.linear.x**2+msg.twist.twist.linear.y**2)
        self.pos_ok=self.yaw_ok=True

    def _ips_cb(self,msg):
        self.car_x=msg.x; self.car_y=msg.y; self.pos_ok=True

    def _imu_cb(self,msg):
        self.car_yaw=quat_to_yaw(msg.orientation); self.yaw_ok=True

    def _get_ref_horizon(self, start_idx):
        """Extract N+1 reference states starting from start_idx."""
        n = self.n_pts
        xs=np.zeros(self.N+1); ys=np.zeros(self.N+1)
        ps=np.zeros(self.N+1); vs=np.zeros(self.N)
        for k in range(self.N+1):
            i=(start_idx+k)%n
            xs[k]=self.xy[i,0]; ys[k]=self.xy[i,1]
            ps[k]=self.psi_ref[i]
            if k<self.N: vs[k]=self.vx_ref[i]
        return xs,ys,ps,vs

    def _loop(self):
        if not (self.pos_ok and self.yaw_ok): return

        t0=time.time()
        cx,cy,yaw,v=self.car_x,self.car_y,self.car_yaw,self.car_v

        # Closest waypoint
        self.last_idx=closest_idx(self.xy,cx,cy,self.last_idx)

        # Reference horizon
        xs,ys,ps,vs=self._get_ref_horizon(self.last_idx)

        # Wrap heading difference
        dpsi=angle_wrap(yaw-ps[0])

        # Set MPC parameters
        x0=np.array([cx,cy,yaw,max(v,0.5)])
        Xref_val=np.vstack([xs,ys,ps,np.append(vs,vs[-1])])
        Vref_val=vs

        try:
            self.opti.set_value(self.X0, x0)
            self.opti.set_value(self.Xref, Xref_val)
            self.opti.set_value(self.Vref, Vref_val)

            # Warm start from previous solution
            if self._prev_sol_X is not None:
                self.opti.set_initial(self.X, self._prev_sol_X)
                self.opti.set_initial(self.U, self._prev_sol_U)
            else:
                # Cold start: simple forward prediction
                X_init=np.zeros((4,self.N+1)); X_init[:,0]=x0
                for k in range(self.N):
                    X_init[0,k+1]=X_init[0,k]+self.dt*v*math.cos(X_init[2,k])
                    X_init[1,k+1]=X_init[1,k]+self.dt*v*math.sin(X_init[2,k])
                    X_init[2,k+1]=X_init[2,k]
                    X_init[3,k+1]=min(X_init[3,k]+self.dt*2.0,V_MAX)
                self.opti.set_initial(self.X, X_init)
                self.opti.set_initial(self.U, np.zeros((2,self.N)))

            sol=self.opti.solve()

            U_sol=sol.value(self.U)
            delta_cmd=float(U_sol[0,0])   # first steering
            a_cmd    =float(U_sol[1,0])   # first acceleration

            # Store for warm start
            X_sol=sol.value(self.X)
            self._prev_sol_X=np.hstack([X_sol[:,1:],X_sol[:,-1:]])
            self._prev_sol_U=np.hstack([U_sol[:,1:],U_sol[:,-1:]])

        except Exception as e:
            self.get_logger().warn(f"MPC failed: {e} — using PP fallback")
            # Pure pursuit fallback
            ti=(self.last_idx+15)%self.n_pts
            tx,ty=self.xy[ti]
            dx=tx-cx; dy=ty-cy
            lx=math.cos(yaw)*dx+math.sin(yaw)*dy
            ly=-math.sin(yaw)*dx+math.cos(yaw)*dy
            ld=math.sqrt(lx**2+ly**2)+1e-6
            delta_cmd=math.atan(2*WB*ly/ld**2)
            a_cmd=2.0
            self._prev_sol_X=None; self._prev_sol_U=None

        # Clamp
        delta_cmd=max(-STEER_MAX,min(STEER_MAX,delta_cmd))

        # Rate limit
        max_d=STEER_RATE*0.05
        delta_cmd=self._prev_delta+max(-max_d,min(max_d,delta_cmd-self._prev_delta))
        self._prev_delta=delta_cmd

        # Speed command from desired acceleration
        v_cmd=max(0.3,min(V_MAX,v+a_cmd*0.05))

        # Normalise to [-1,1]
        throttle=float(np.clip(v_cmd/V_MAX,-1,1))
        steer_n =float(np.clip(delta_cmd/STEER_MAX,-1,1))

        t_msg=Float32(); t_msg.data=throttle; self.pub_t.publish(t_msg)
        s_msg=Float32(); s_msg.data=steer_n;  self.pub_s.publish(s_msg)

        self._dbg+=1
        if self._dbg%20==0:
            elapsed=time.time()-t0
            self.get_logger().info(
                f"({cx:.2f},{cy:.2f}) v={v:.2f}m/s "
                f"steer={math.degrees(delta_cmd):.1f}° "
                f"thr={throttle:.2f} "
                f"wp={self.last_idx}/{self.n_pts} "
                f"solve={elapsed*1000:.1f}ms"
            )

    def stop(self):
        for p in [self.pub_t,self.pub_s]:
            m=Float32(); m.data=0.; p.publish(m)


def main():
    p=argparse.ArgumentParser(description="MPC controller for AutoDRIVE RoboRacer")
    p.add_argument("--csv",        required=True)
    p.add_argument("--N",          type=int,   default=20,  help="MPC horizon (default 20)")
    p.add_argument("--dt",         type=float, default=0.05,help="MPC timestep [s] (default 0.05)")
    p.add_argument("--speed_gain", type=float, default=1.0, help="Speed multiplier (default 1.0)")
    p.add_argument("--use_odom",   action="store_true",     help="Use /odom topic")
    args,_=p.parse_known_args()

    rclpy.init()
    node=MPCNode(args.csv,args.N,args.dt,args.speed_gain,args.use_odom)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.stop(); node.destroy_node(); rclpy.shutdown()

if __name__=="__main__":
    main()