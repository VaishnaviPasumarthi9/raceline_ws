# ============================================================
#  COMPLETE EXECUTION GUIDE
#  Platform: Windows WSL + ROS2 Jazzy
# ============================================================


# ────────────────────────────────────────────────────────────
# STEP 0  —  EXPECTED FOLDER STRUCTURE
# ────────────────────────────────────────────────────────────
#
# Before you start, your working folder should look like this:
#
#  ~/raceline_ws/
#  ├── maps/
#  │   └── your_map.pbstream          ← your Cartographer map file
#  ├── raceline_generator.py          ← main algorithm (from Claude)
#  ├── raceline_rviz_publisher.py     ← RViz publisher (from Claude)
#  ├── raceline.rviz                  ← RViz config  (from Claude)
#  ├── requirements.txt               ← Python deps  (from Claude)
#  └── output/                        ← created automatically
#      ├── raceline.csv               ← generated after Step 3
#      └── raceline.png               ← generated after Step 3


# ────────────────────────────────────────────────────────────
# STEP 1  —  CREATE THE WORKSPACE
# ────────────────────────────────────────────────────────────

mkdir -p ~/raceline_ws/maps
mkdir -p ~/raceline_ws/output
cd ~/raceline_ws

# Copy your pbstream into the maps folder.
# From Windows Explorer: paste your file into
#   \\wsl$\Ubuntu\home\<your_username>\raceline_ws\maps\
# Then rename it to:  your_map.pbstream
# Verify it's there:
ls maps/


# ────────────────────────────────────────────────────────────
# STEP 2  —  INSTALL CARTOGRAPHER (check first)
# ────────────────────────────────────────────────────────────

# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# Check if Cartographer is already installed
ros2 pkg list | grep cartographer

# If you see   cartographer_ros   →  already installed, skip apt install below
# If you see nothing              →  run this:
sudo apt update
sudo apt install -y ros-jazzy-cartographer ros-jazzy-cartographer-ros


# ────────────────────────────────────────────────────────────
# STEP 3  —  CONVERT pbstream → pgm + yaml
# ────────────────────────────────────────────────────────────

cd ~/raceline_ws

source /opt/ros/jazzy/setup.bash

# This command reads the pbstream and writes:
#   maps/my_map.pgm   (greyscale occupancy image)
#   maps/my_map.yaml  (resolution, origin metadata)
ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
  -pbstream_filename maps/your_map.pbstream \
  -map_filestem maps/my_map \
  -resolution 0.05

# Verify output:
ls maps/
# You should see:  my_map.pgm   my_map.yaml   your_map.pbstream


# ────────────────────────────────────────────────────────────
# STEP 4  —  INSTALL PYTHON DEPENDENCIES
# ────────────────────────────────────────────────────────────

cd ~/raceline_ws

# Install pip if not already present
sudo apt install -y python3-pip

# Install all required Python packages
pip3 install numpy scipy opencv-python scikit-image matplotlib PyYAML quadprog

# Verify quadprog (the QP solver) installed correctly
python3 -c "import quadprog; print('quadprog OK')"
# Should print:  quadprog OK
# If it fails, try:  pip3 install quadprog --no-build-isolation


# ────────────────────────────────────────────────────────────
# STEP 5  —  RUN THE RACELINE GENERATOR
# ────────────────────────────────────────────────────────────

cd ~/raceline_ws

python3 raceline_generator.py \
  --pgm  maps/my_map.pgm \
  --yaml maps/my_map.yaml \
  --output output/ \
  --step   0.1 \
  --safety 0.1 \
  --iter   3

# ── What you will see ──────────────────────────────────────
# [1/7] Loading map...
#       Grid: (480, 640)  @  0.05 m/px
# [2/7] Pre-processing occupancy grid...
# [3/7] Extracting centreline via skeletonisation...
#       1234 skeleton points found
# [4/7] Computing track widths...
# [5/7] Smoothing & resampling centreline...
#       Resampled to 856 points at 0.1 m intervals
# [6/7] Minimum-curvature QP optimisation (backend=quadprog, 3 iterations)...
#   [QP iter 1/3] max Δα = 0.1234 m
#   [QP iter 2/3] max Δα = 0.0021 m
#   [QP iter 3/3] max Δα = 0.0003 m
#   Converged.
# [7/7] Computing curvature and velocity profile...
#       Estimated lap time: 18.43 s
#       Max speed: 8.00 m/s   Min speed: 1.23 m/s
# [output] Raceline CSV written: output/raceline.csv  (856 points)
# [output] Visualisation saved:  output/raceline.png

# Open the PNG to verify the raceline looks correct:
# (view from Windows: \\wsl$\Ubuntu\home\<user>\raceline_ws\output\raceline.png)


# ────────────────────────────────────────────────────────────
# STEP 6  —  VISUALISE IN RViz
# ────────────────────────────────────────────────────────────
# Open 3 separate WSL terminals (or use tmux)

# ── Terminal 1: start the map server ──────────────────────
source /opt/ros/jazzy/setup.bash
cd ~/raceline_ws
ros2 run nav2_map_server map_server \
  --ros-args -p yaml_filename:=maps/my_map.yaml \
             -p use_sim_time:=false

# If nav2_map_server is not installed:
# sudo apt install -y ros-jazzy-nav2-map-server

# ── Terminal 2: publish the raceline markers ───────────────
source /opt/ros/jazzy/setup.bash
cd ~/raceline_ws
python3 raceline_rviz_publisher.py \
  --csv output/raceline.csv \
  --frame map \
  --loop

# ── Terminal 3: open RViz ──────────────────────────────────
source /opt/ros/jazzy/setup.bash
cd ~/raceline_ws
rviz2 -d raceline.rviz

# ── In RViz ───────────────────────────────────────────────
# If the config loads correctly you will immediately see:
#   - Grey occupancy grid (your track)
#   - Orange line  = fastest raceline
#   - Coloured dots: Green=fast  Yellow=medium  Red=slow corners
#   - Blue line = left track boundary
#   - Red line  = right track boundary
#   - White text labels showing speed in m/s
#
# If nothing shows:
#   1. Check Fixed Frame (top left) is set to "map"
#   2. Click Add → By Topic and manually add /raceline/markers


# ────────────────────────────────────────────────────────────
# TROUBLESHOOTING
# ────────────────────────────────────────────────────────────

# Problem: "No module named 'cv2'"
pip3 install opencv-python

# Problem: quadprog build fails
sudo apt install -y python3-dev build-essential
pip3 install quadprog

# Problem: skeletonisation too sparse / raceline looks wrong
# → Try increasing dilation:
python3 raceline_generator.py --pgm maps/my_map.pgm --yaml maps/my_map.yaml \
  --output output/ --dilation 5 --safety 0.15

# Problem: RViz shows nothing / fixed frame error
# → Make sure map_server is running (Terminal 1) before opening RViz
# → Set Fixed Frame = "map" in RViz Global Options

# Problem: cartographer_pbstream_to_ros_map not found
ros2 pkg executables cartographer_ros
# Lists all available executables in the package
