# vacuum_driver

Minimal ROS 2 package for Webots communication only.

Provided node:

- `pure_driver` (`vacuum_driver.pure_driver`)

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --base-paths src/vacuum_driver --packages-select vacuum_driver --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 run vacuum_driver pure_driver
```
