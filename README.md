# Minimum-Curvature Raceline Generator
### From Cartographer `.pbstream` → fastest solo raceline

Implements the **Geometric Minimum-Curvature Optimisation** algorithm used by the
[ForzaETH Race Stack](https://github.com/ForzaETH/race_stack) and formalised in:

> Heilmeier et al. (2020). *Minimum curvature trajectory planning and control
> for an autonomous race car.* Vehicle System Dynamics, 58(10).

---

## Pipeline Overview

```
pbstream / pgm
     │
     ▼
[1] Load occupancy grid
     │
     ▼
[2] Pre-process  (morphological clean, largest component)
     │
     ▼
[3] Skeletonise → raw centreline points
     │
     ▼
[4] Track width sampling along normals
     │
     ▼
[5] Smooth & resample to uniform arc-length steps
     │
     ▼
[6] Iterative QP minimum-curvature optimisation   ← CORE ALGORITHM
     │    minimise ∑ κᵢ²  subject to  -w_right ≤ αᵢ ≤ w_left
     ▼
[7] Forward–backward velocity profile solver
     │
     ▼
raceline.csv  +  raceline.png
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **QP solver:** `quadprog` is recommended and installs with pip.
> If it fails on your platform, install `osqp` instead — the code auto-detects.

---

## Usage

### Option A — Direct `.pbstream` input (needs Cartographer or protobuf)

```bash
# If cartographer_pbstream_to_ros_map is on your PATH (ROS install):
python raceline_generator.py --pbstream path/to/map.pbstream --output output/

# If Cartographer is NOT installed, the script will try direct protobuf parsing.
# Install the protobuf package first:
pip install protobuf
python raceline_generator.py --pbstream path/to/map.pbstream
```

### Option B — Pre-exported `.pgm` + `.yaml` (recommended for ROS-less environments)

Export from ROS once:
```bash
rosrun map_server map_saver -f my_map
# or with Cartographer:
cartographer_pbstream_to_ros_map \
    -pbstream_filename map.pbstream \
    -map_filestem my_map
```

Then run offline:
```bash
python raceline_generator.py --pgm my_map.pgm --yaml my_map.yaml --output output/
```

### Option C — PNG/PGM without YAML (will use default 0.05 m/px resolution)
```bash
python raceline_generator.py --pgm my_map.png --resolution 0.05 --output output/
```

---

## Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--step` | `0.1` | Raceline sample spacing [m] |
| `--safety` | `0.1` | Safety margin from walls [m] |
| `--iter` | `3` | QP iterations (more = less linearisation error) |
| `--dilation` | `3` | Wall inward erosion [px] — increase for wider safety margin |
| `--v_max` | `8.0` | Vehicle max speed [m/s] |
| `--ax_max` | `6.0` | Max longitudinal acceleration [m/s²] |
| `--ax_min` | `-8.0` | Max braking deceleration [m/s²] |
| `--ay_max` | `7.0` | Max lateral acceleration [m/s²] |

---

## Output Files

### `raceline.csv`
```
x_m, y_m, w_tr_right_m, w_tr_left_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
```
Compatible with the ForzaETH race_stack CSV format for direct use in the stack's
pure pursuit / MPCC controller.

### `raceline.png`
Two panels:
- **Left:** Occupancy grid with centreline (cyan dashed) and optimised raceline (red)
- **Right:** Raceline colour-coded by velocity (green = fast, red = slow)

---

## Algorithm Details

### Minimum-Curvature QP (Heilmeier 2020)

The optimisation variable is **α ∈ ℝᴺ**, the lateral offset of each raceline
point from the reference centreline:

```
p_i = ref_i + α_i · n̂_i
```

The squared curvature at each point is approximated via the 2nd finite
difference:

```
κ_i ≈ (p_{i-1} - 2p_i + p_{i+1}) / Δs²
```

This is quadratic in α, giving the QP:

```
minimise   αᵀ H α + fᵀ α
subject to  -w_right_i ≤ α_i ≤ w_left_i   ∀i
```

**Iterative invocation:** The reference line is updated to the previous
solution after each QP solve, reducing linearisation error in sharp corners
(Heilmeier's key contribution). Typically 3 iterations suffice.

### Velocity Profile

Given the optimised curvature κ(s):
1. Corner-speed limit: `v_corner = √(ay_max / |κ|)`
2. Forward pass: acceleration-limited integration
3. Backward pass: braking-limited integration
4. Final speed: `vx = min(v_forward, v_backward, v_corner)`

---

## Tuning for Your Car

Edit the `VehicleParams` class in `raceline_generator.py` or pass CLI flags:

```python
vp.v_max_mps    = 8.0     # top speed
vp.ax_max_mps2  = 6.0     # peak acceleration
vp.ax_min_mps2  = -8.0    # peak braking
vp.ay_max_mps2  = 7.0     # peak cornering (depends on tyre mu)
vp.mu           = 0.8     # friction coefficient
```

For F1TENTH (1:10 scale) these defaults are reasonable.
For a full-size kart or formula car, scale accordingly.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No QP solver` warning | `pip install quadprog` |
| Skeleton too sparse | Increase `--dilation` or check map quality |
| Raceline cuts corners too aggressively | Increase `--safety` |
| `Cannot load pbstream` | Pre-export to `.pgm` with `cartographer_pbstream_to_ros_map` |
| Map resolution wrong | Pass `--resolution 0.05` (metres per pixel) |
