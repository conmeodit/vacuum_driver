# vacuum_driver

ROS 2 package for the Webots vacuum robot.

Provided nodes:

- `pure_driver`: Webots bridge for `/scan`, `/odom`, TF and `/cmd_vel`.
- `slam_session_manager_node`: resets `slam_toolbox` at launch startup.
- `autonomous_cleaning_node`: frontier exploration followed by visited-cell coverage.

## Build

```bash
rm -rf ~/ros2_ws/build/vacuum_driver ~/ros2_ws/install/vacuum_driver
cd ~/ros2_ws
colcon build --base-paths src/vacuum_driver --packages-select vacuum_driver --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 run vacuum_driver pure_driver
```

Mapping only:

```bash
ros2 launch vacuum_driver mapping.launch.py
```

Autonomous mapping and full reachable-area coverage:

```bash
ros2 launch vacuum_driver autonomous_mapping.launch.py
```

RViz topics used by the autonomous node:

- `/autonomy/path`: current A* path.
- `/autonomy/markers`: robot body, footprint, target, frontier cells and visited cells.
