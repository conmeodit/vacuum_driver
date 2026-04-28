import math
import os
import sys
from typing import List, Optional

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

# Defaults extracted from worlds/xe.wbt
WEBOTS_ROBOT_NAME = 'vacuum_robot'
WEBOTS_LIDAR_NAME = 'lidar'
WEBOTS_IMU_NAME = 'MPU-9250'

WEBOTS_WHEEL_RADIUS_M = 0.0425
WEBOTS_WHEEL_OFFSET_X_M = 0.1
WEBOTS_WHEEL_OFFSET_Y_M = 0.21
WEBOTS_WHEELBASE_M = WEBOTS_WHEEL_OFFSET_X_M * 2.0
WEBOTS_WHEEL_SEPARATION_M = WEBOTS_WHEEL_OFFSET_Y_M * 2.0

WEBOTS_LIDAR_OFFSET_X_M = -0.1
WEBOTS_LIDAR_OFFSET_Y_M = 0.0
WEBOTS_LIDAR_OFFSET_Z_M = 0.081

WEBOTS_LEFT_MOTOR_NAMES = ('motor-1', 'motor-3')
WEBOTS_RIGHT_MOTOR_NAMES = ('motor-2', 'motor-4')
WEBOTS_LEFT_ENCODER_NAMES = ('enc-3',)
WEBOTS_RIGHT_ENCODER_NAMES = ('enc-4',)

DEFAULT_CMD_VEL_TOPIC = '/cmd_vel'
DEFAULT_CMD_TIMEOUT_SEC = 0.8
DEFAULT_MAX_LINEAR_SPEED_MPS = 0.15
DEFAULT_MAX_ANGULAR_SPEED_RADPS = 0.8


def _as_string_list(value, fallback=None):
    if isinstance(value, str):
        parsed = [item.strip() for item in value.split(',') if item.strip()]
    elif isinstance(value, (list, tuple)):
        parsed = [str(item).strip() for item in value if str(item).strip()]
    else:
        parsed = []
    if parsed:
        return parsed
    return list(fallback) if fallback is not None else []


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _quat_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def _init_webots_env():
    if 'WEBOTS_CONTROLLER_URL' not in os.environ:
        # Default to localhost for native Linux. If running in WSL,
        # resolve the Windows host IP through the default route.
        windows_ip = ''
        try:
            with open('/proc/version', 'r', encoding='utf-8') as version_file:
                version_text = version_file.read().lower()
            if 'microsoft' in version_text:
                windows_ip = os.popen(
                    "ip route show default | awk '{print $3}'"
                ).read().strip()
        except Exception:
            windows_ip = ''
        if not windows_ip:
            windows_ip = '127.0.0.1'
        os.environ['WEBOTS_CONTROLLER_URL'] = (
            f"tcp://{windows_ip}:1234/{WEBOTS_ROBOT_NAME}"
        )
    if 'WEBOTS_HOME' not in os.environ:
        os.environ['WEBOTS_HOME'] = '/usr/local/webots'
    controller_path = '/usr/local/webots/lib/controller/python'
    if controller_path not in sys.path:
        sys.path.append(controller_path)


_init_webots_env()
from controller import Robot


class PureWebotsDriver(Node):
    def __init__(self):
        super().__init__('pure_webots_driver')

        self.declare_parameter('use_encoder_odom', True)
        self.declare_parameter('use_open_loop_odom', False)
        self.declare_parameter('publish_odom_tf', True)
        self.declare_parameter('publish_lidar_tf', True)
        self.declare_parameter('reverse_scan', True)
        self.declare_parameter('lidar_yaw_180', False)
        self.declare_parameter('scan_use_inf_for_max_range', True)
        self.declare_parameter('scan_max_range_margin', 0.02)
        self.declare_parameter('scan_filter_enabled', True)
        self.declare_parameter('scan_filter_window', 5)
        self.declare_parameter('scan_filter_min_valid_neighbors', 3)
        self.declare_parameter('scan_filter_outlier_threshold_m', 0.12)
        self.declare_parameter('scan_filter_temporal_alpha', 0.60)
        self.declare_parameter('scan_filter_temporal_jump_threshold_m', 0.20)

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('scan_raw_topic', '/scan/raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_raw_topic', '/odom/raw')
        self.declare_parameter('cmd_vel_topic', DEFAULT_CMD_VEL_TOPIC)
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('imu_raw_topic', '/imu/data_raw')

        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('lidar_frame_id', 'laser')
        self.declare_parameter('imu_frame_id', 'base_link')

        self.declare_parameter('lidar_device_name', WEBOTS_LIDAR_NAME)
        self.declare_parameter('imu_device_name', WEBOTS_IMU_NAME)
        self.declare_parameter('imu_gyro_device_name', '')
        self.declare_parameter('imu_accel_device_name', '')

        self.declare_parameter('lidar_offset_x', WEBOTS_LIDAR_OFFSET_X_M)
        self.declare_parameter('lidar_offset_y', WEBOTS_LIDAR_OFFSET_Y_M)
        self.declare_parameter('lidar_offset_z', WEBOTS_LIDAR_OFFSET_Z_M)

        self.declare_parameter('wheel_radius', WEBOTS_WHEEL_RADIUS_M)
        self.declare_parameter('wheel_separation', WEBOTS_WHEEL_SEPARATION_M)
        self.declare_parameter('wheelbase', WEBOTS_WHEELBASE_M)
        self.declare_parameter('left_motor_names', list(WEBOTS_LEFT_MOTOR_NAMES))
        self.declare_parameter('right_motor_names', list(WEBOTS_RIGHT_MOTOR_NAMES))
        self.declare_parameter('left_encoder_names', list(WEBOTS_LEFT_ENCODER_NAMES))
        self.declare_parameter('right_encoder_names', list(WEBOTS_RIGHT_ENCODER_NAMES))

        self.declare_parameter('odom_angular_scale', 1.0)
        self.declare_parameter('reject_encoder_jump', True)
        self.declare_parameter('max_wheel_step_delta_m', 0.08)
        self.declare_parameter('enable_imu', True)
        self.declare_parameter('use_imu_for_odom', False)
        self.declare_parameter('allow_gyro_imu_for_odom', False)
        self.declare_parameter('imu_yaw_blend', 0.35)
        self.declare_parameter('cmd_vel_timeout_sec', DEFAULT_CMD_TIMEOUT_SEC)
        self.declare_parameter('max_linear_speed_mps', DEFAULT_MAX_LINEAR_SPEED_MPS)
        self.declare_parameter('max_angular_speed_radps', DEFAULT_MAX_ANGULAR_SPEED_RADPS)

        self.use_encoder_odom = bool(self.get_parameter('use_encoder_odom').value)
        self.use_open_loop_odom = bool(self.get_parameter('use_open_loop_odom').value)
        self.publish_odom_tf = bool(self.get_parameter('publish_odom_tf').value)
        self.publish_lidar_tf_enabled = bool(self.get_parameter('publish_lidar_tf').value)
        self.reverse_scan = bool(self.get_parameter('reverse_scan').value)
        self.lidar_yaw_180 = bool(self.get_parameter('lidar_yaw_180').value)
        self.scan_use_inf_for_max_range = bool(
            self.get_parameter('scan_use_inf_for_max_range').value
        )
        self.scan_max_range_margin = max(
            0.0, float(self.get_parameter('scan_max_range_margin').value)
        )
        self.scan_filter_enabled = bool(
            self.get_parameter('scan_filter_enabled').value
        )
        self.scan_filter_window = max(
            1, int(self.get_parameter('scan_filter_window').value)
        )
        if (self.scan_filter_window % 2) == 0:
            self.scan_filter_window += 1
        self.scan_filter_min_valid_neighbors = max(
            1, int(self.get_parameter('scan_filter_min_valid_neighbors').value)
        )
        self.scan_filter_outlier_threshold_m = max(
            0.01,
            float(self.get_parameter('scan_filter_outlier_threshold_m').value),
        )
        self.scan_filter_temporal_alpha = _clamp(
            float(self.get_parameter('scan_filter_temporal_alpha').value), 0.0, 1.0
        )
        self.scan_filter_temporal_jump_threshold_m = max(
            0.02,
            float(self.get_parameter('scan_filter_temporal_jump_threshold_m').value),
        )

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.scan_raw_topic = str(self.get_parameter('scan_raw_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.odom_raw_topic = str(self.get_parameter('odom_raw_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.imu_topic = str(self.get_parameter('imu_topic').value)
        self.imu_raw_topic = str(self.get_parameter('imu_raw_topic').value)

        self.odom_frame_id = str(self.get_parameter('odom_frame_id').value)
        self.base_frame_id = str(self.get_parameter('base_frame_id').value)
        self.lidar_frame_id = str(self.get_parameter('lidar_frame_id').value)
        self.imu_frame_id = str(self.get_parameter('imu_frame_id').value)

        self.lidar_device_name = str(self.get_parameter('lidar_device_name').value).strip()
        self.imu_device_name = str(self.get_parameter('imu_device_name').value).strip()
        self.imu_gyro_device_name = str(
            self.get_parameter('imu_gyro_device_name').value
        ).strip()
        self.imu_accel_device_name = str(
            self.get_parameter('imu_accel_device_name').value
        ).strip()

        self.lidar_offset_x = float(self.get_parameter('lidar_offset_x').value)
        self.lidar_offset_y = float(self.get_parameter('lidar_offset_y').value)
        self.lidar_offset_z = float(self.get_parameter('lidar_offset_z').value)

        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.wheel_separation = float(self.get_parameter('wheel_separation').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.left_motor_names = _as_string_list(
            self.get_parameter('left_motor_names').value,
            fallback=WEBOTS_LEFT_MOTOR_NAMES,
        )
        self.right_motor_names = _as_string_list(
            self.get_parameter('right_motor_names').value,
            fallback=WEBOTS_RIGHT_MOTOR_NAMES,
        )
        self.left_encoder_names = _as_string_list(
            self.get_parameter('left_encoder_names').value,
            fallback=WEBOTS_LEFT_ENCODER_NAMES,
        )
        self.right_encoder_names = _as_string_list(
            self.get_parameter('right_encoder_names').value,
            fallback=WEBOTS_RIGHT_ENCODER_NAMES,
        )

        self.odom_angular_scale = float(self.get_parameter('odom_angular_scale').value)
        self.reject_encoder_jump = bool(
            self.get_parameter('reject_encoder_jump').value
        )
        self.max_wheel_step_delta_m = max(
            0.01, float(self.get_parameter('max_wheel_step_delta_m').value)
        )
        self.enable_imu = bool(self.get_parameter('enable_imu').value)
        self.use_imu_for_odom = bool(self.get_parameter('use_imu_for_odom').value)
        self.allow_gyro_imu_for_odom = bool(
            self.get_parameter('allow_gyro_imu_for_odom').value
        )
        self.imu_yaw_blend = _clamp(
            float(self.get_parameter('imu_yaw_blend').value), 0.0, 1.0
        )
        self.cmd_vel_timeout_sec = max(
            0.1, float(self.get_parameter('cmd_vel_timeout_sec').value)
        )
        self.max_linear_speed_mps = max(
            0.01, float(self.get_parameter('max_linear_speed_mps').value)
        )
        self.max_angular_speed_radps = max(
            0.05, float(self.get_parameter('max_angular_speed_radps').value)
        )

        self.current_v = 0.0
        self.current_w = 0.0
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.last_cmd_ns = int(self.get_clock().now().nanoseconds)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.x_raw = 0.0
        self.y_raw = 0.0
        self.yaw_raw = 0.0
        self.prev_left_pos = None
        self.prev_right_pos = None

        self.imu_mode = 'none'
        self.imu_inertial = None
        self.imu_gyro = None
        self.imu_accel = None
        self.last_imu_time_sec = None
        self.last_imu_rpy = None
        self.last_imu_linear_accel = (0.0, 0.0, 0.0)
        self.last_imu_angular_velocity = (0.0, 0.0, 0.0)
        self.last_imu_delta_yaw = 0.0
        self.encoder_jump_counter = 0
        self.prev_filtered_ranges: Optional[List[float]] = None

        self.get_logger().info(
            f"Connecting Webots at: {os.environ.get('WEBOTS_CONTROLLER_URL', '(unset)')}"
        )
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # Build a monotonic ROS stamp stream anchored to current wall time,
        # while progressing by Webots simulation time to avoid wall-clock jumps.
        sim_now = float(self.robot.getTime())
        self.stamp_wall_base_ns = int(self.get_clock().now().nanoseconds)
        self.stamp_sim_base_sec = sim_now
        self.last_stamp_ns = self.stamp_wall_base_ns

        self.left_motors = self._get_devices_by_names(self.left_motor_names, side='left')
        self.right_motors = self._get_devices_by_names(
            self.right_motor_names, side='right'
        )
        for motor in self.left_motors + self.right_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        self.max_motor_speed = self._motor_velocity_limit()

        self.left_position_sensors = []
        self.right_position_sensors = []
        if self.use_encoder_odom:
            self.left_position_sensors = self._init_named_position_sensors(
                self.left_encoder_names
            )
            self.right_position_sensors = self._init_named_position_sensors(
                self.right_encoder_names
            )
            if not self.left_position_sensors or not self.right_position_sensors:
                self.get_logger().warn(
                    "Encoder odometry requested but encoders are incomplete. "
                    "Falling back to open-loop odometry."
                )
                self.use_encoder_odom = False
                self.use_open_loop_odom = True

        self.lidar, lidar_name = self._try_get_device([self.lidar_device_name])
        if self.lidar is None:
            raise RuntimeError(f'Lidar device "{self.lidar_device_name}" not found')
        self.lidar.enable(self.time_step)
        self.lidar_fov = float(self.lidar.getFov())
        self.lidar_resolution = int(self.lidar.getHorizontalResolution())
        self.lidar_min_range = float(self.lidar.getMinRange())
        self.lidar_max_range = float(self.lidar.getMaxRange())
        if self.lidar_resolution > 1:
            self.lidar_angle_increment = self.lidar_fov / (self.lidar_resolution - 1)
        else:
            self.lidar_angle_increment = 0.0
        self.lidar_angle_min = -self.lidar_fov / 2.0
        self.lidar_angle_max = self.lidar_fov / 2.0

        if self.enable_imu:
            self._init_imu_devices()

        if self.use_imu_for_odom and self.imu_mode == 'gyro_accel':
            if self.allow_gyro_imu_for_odom:
                self.get_logger().warn(
                    'Using gyro_accel IMU fusion for odom yaw. '
                    'This can drift over time.'
                )
            else:
                self.get_logger().warn(
                    'IMU yaw fusion disabled for odom because only gyro_accel mode '
                    'is available. Raw gyro integration was causing long-term odom drift.'
                )
                self.use_imu_for_odom = False

        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)
        self.scan_raw_pub = self.create_publisher(LaserScan, self.scan_raw_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.odom_raw_pub = self.create_publisher(Odometry, self.odom_raw_topic, 10)
        self.imu_pub = self.create_publisher(Imu, self.imu_topic, 10)
        self.imu_raw_pub = self.create_publisher(Imu, self.imu_raw_topic, 10)
        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self._cmd_vel_callback, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        if self.publish_lidar_tf_enabled:
            self._publish_lidar_static_tf()
        self.timer = self.create_timer(self.dt, self.step_callback)

        self.get_logger().info(
            "pure_driver bridge ready: "
            f"lidar={lidar_name}, imu_mode={self.imu_mode}, "
            f"motors(L={self.left_motor_names},R={self.right_motor_names}), "
            f"encoders(L={self.left_encoder_names},R={self.right_encoder_names}), "
            f"cmd_vel_topic={self.cmd_vel_topic}, "
            f"odom_mode={'encoder' if self.use_encoder_odom else ('open_loop' if self.use_open_loop_odom else 'static')}, "
            f"imu_yaw_fusion={'on' if self.use_imu_for_odom else 'off'}, "
            f"scan_filter={'on' if self.scan_filter_enabled else 'off'}"
        )

    def _cmd_vel_callback(self, msg):
        self.cmd_linear = _clamp(
            float(msg.linear.x), -self.max_linear_speed_mps, self.max_linear_speed_mps
        )
        self.cmd_angular = _clamp(
            float(msg.angular.z),
            -self.max_angular_speed_radps,
            self.max_angular_speed_radps,
        )
        self.last_cmd_ns = int(self.get_clock().now().nanoseconds)

    def _try_get_device(self, names):
        if not names:
            return None, None
        for name in names:
            try:
                device = self.robot.getDevice(name)
            except Exception:
                device = None
            if device is not None:
                return device, name
        return None, None

    def _get_devices_by_names(self, names, side):
        devices = []
        missing = []
        for name in names:
            device, _ = self._try_get_device([name])
            if device is None:
                missing.append(name)
            else:
                devices.append(device)
        if missing:
            raise RuntimeError(
                f'Missing {side} motor device(s): {", ".join(missing)}'
            )
        return devices

    def _init_named_position_sensors(self, names):
        sensors = []
        for name in names:
            sensor, _ = self._try_get_device([name])
            if sensor is None or not hasattr(sensor, 'getValue'):
                continue
            try:
                sensor.enable(self.time_step)
            except Exception:
                continue
            sensors.append(sensor)
        return sensors

    def _motor_velocity_limit(self):
        limits = []
        for motor in self.left_motors + self.right_motors:
            try:
                speed = float(motor.getMaxVelocity())
            except Exception:
                speed = 0.0
            if speed > 0.0 and math.isfinite(speed):
                limits.append(speed)
        if not limits:
            return None
        return min(limits)

    def _init_imu_devices(self):
        inertial_names = []
        if self.imu_device_name:
            inertial_names.append(self.imu_device_name)
        inertial_names.extend(['MPU-9250', 'mpu-9250', 'imu'])
        imu_dev, imu_name = self._try_get_device(inertial_names)
        if imu_dev is not None and hasattr(imu_dev, 'getRollPitchYaw'):
            imu_dev.enable(self.time_step)
            self.imu_inertial = imu_dev
            self.imu_mode = 'inertial'
            self.get_logger().info(f'IMU inertial mode enabled: {imu_name}')
            return

        gyro_names = []
        if self.imu_gyro_device_name:
            gyro_names.append(self.imu_gyro_device_name)
        gyro_names.extend(
            ['MPU-9250 gyro', 'mpu-9250 gyro', 'gyro', 'gyroscope', 'imu_gyro']
        )
        accel_names = []
        if self.imu_accel_device_name:
            accel_names.append(self.imu_accel_device_name)
        accel_names.extend(
            [
                'MPU-9250 accelerometer',
                'mpu-9250 accelerometer',
                'accelerometer',
                'accel',
                'imu_accel',
            ]
        )

        gyro_dev, gyro_name = self._try_get_device(gyro_names)
        accel_dev, accel_name = self._try_get_device(accel_names)
        if gyro_dev is not None and accel_dev is not None:
            gyro_dev.enable(self.time_step)
            accel_dev.enable(self.time_step)
            self.imu_gyro = gyro_dev
            self.imu_accel = accel_dev
            self.imu_mode = 'gyro_accel'
            self.get_logger().info(
                f'IMU gyro+accel mode enabled: gyro={gyro_name}, accel={accel_name}'
            )
            return

        self.imu_mode = 'none'
        self.get_logger().warn('IMU not found. /imu topics will not be published.')

    def _read_vec3(self, device):
        if device is None:
            return None
        try:
            values = device.getValues()
        except Exception:
            return None
        if values is None or len(values) < 3:
            return None
        x = float(values[0])
        y = float(values[1])
        z = float(values[2])
        if (
            math.isnan(x)
            or math.isnan(y)
            or math.isnan(z)
            or math.isinf(x)
            or math.isinf(y)
            or math.isinf(z)
        ):
            return None
        return x, y, z

    def _read_rpy(self, device):
        if device is None or not hasattr(device, 'getRollPitchYaw'):
            return None
        try:
            values = device.getRollPitchYaw()
        except Exception:
            return None
        if values is None or len(values) < 3:
            return None
        roll = float(values[0])
        pitch = float(values[1])
        yaw = float(values[2])
        if (
            math.isnan(roll)
            or math.isnan(pitch)
            or math.isnan(yaw)
            or math.isinf(roll)
            or math.isinf(pitch)
            or math.isinf(yaw)
        ):
            return None
        return roll, pitch, yaw

    def _mean_sensor_value(self, sensors):
        if not sensors:
            return None
        values = []
        for sensor in sensors:
            try:
                value = float(sensor.getValue())
            except Exception:
                continue
            if math.isfinite(value):
                values.append(value)
        if not values:
            return None
        return sum(values) / float(len(values))

    def _apply_cmd(self):
        cmd_age_sec = (
            int(self.get_clock().now().nanoseconds) - self.last_cmd_ns
        ) * 1e-9
        if cmd_age_sec > self.cmd_vel_timeout_sec:
            linear = 0.0
            angular = 0.0
        else:
            linear = self.cmd_linear
            angular = self.cmd_angular

        left_mps = linear - 0.5 * angular * self.wheel_separation
        right_mps = linear + 0.5 * angular * self.wheel_separation
        left_rad_s = left_mps / max(self.wheel_radius, 1e-6)
        right_rad_s = right_mps / max(self.wheel_radius, 1e-6)

        if self.max_motor_speed is not None:
            left_rad_s = _clamp(left_rad_s, -self.max_motor_speed, self.max_motor_speed)
            right_rad_s = _clamp(
                right_rad_s, -self.max_motor_speed, self.max_motor_speed
            )

        self.current_v = 0.5 * (left_rad_s + right_rad_s) * self.wheel_radius
        self.current_w = (
            (right_rad_s - left_rad_s) * self.wheel_radius
        ) / max(self.wheel_separation, 1e-6)

        for motor in self.left_motors:
            motor.setVelocity(left_rad_s)
        for motor in self.right_motors:
            motor.setVelocity(right_rad_s)

    def _publish_lidar_static_tf(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self._next_stamp()
        tf_msg.header.frame_id = self.base_frame_id
        tf_msg.child_frame_id = self.lidar_frame_id
        tf_msg.transform.translation.x = self.lidar_offset_x
        tf_msg.transform.translation.y = self.lidar_offset_y
        tf_msg.transform.translation.z = self.lidar_offset_z
        if self.lidar_yaw_180:
            tf_msg.transform.rotation.x = 0.0
            tf_msg.transform.rotation.y = 0.0
            tf_msg.transform.rotation.z = 1.0
            tf_msg.transform.rotation.w = 0.0
        else:
            tf_msg.transform.rotation.x = 0.0
            tf_msg.transform.rotation.y = 0.0
            tf_msg.transform.rotation.z = 0.0
            tf_msg.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(tf_msg)

    def _publish_odom(self, stamp):
        linear_vel = 0.0
        angular_vel = 0.0

        if self.use_encoder_odom:
            left = self._mean_sensor_value(self.left_position_sensors)
            right = self._mean_sensor_value(self.right_position_sensors)
            if left is not None and right is not None:
                if self.prev_left_pos is not None and self.prev_right_pos is not None:
                    delta_left = (left - self.prev_left_pos) * self.wheel_radius
                    delta_right = (right - self.prev_right_pos) * self.wheel_radius
                    if self.reject_encoder_jump and (
                        abs(delta_left) > self.max_wheel_step_delta_m
                        or abs(delta_right) > self.max_wheel_step_delta_m
                    ):
                        self.encoder_jump_counter += 1
                        if self.encoder_jump_counter <= 3 or (
                            self.encoder_jump_counter % 50 == 0
                        ):
                            self.get_logger().warn(
                                "Encoder jump rejected: "
                                f"dl={delta_left:.3f}m, dr={delta_right:.3f}m "
                                f"(threshold={self.max_wheel_step_delta_m:.3f}m)"
                            )
                        self.prev_left_pos = left
                        self.prev_right_pos = right
                        return
                    delta_s = 0.5 * (delta_left + delta_right)
                    delta_yaw_enc = (delta_right - delta_left) / max(
                        self.wheel_separation, 1e-6
                    )
                    delta_yaw_raw = delta_yaw_enc * self.odom_angular_scale
                    delta_yaw = delta_yaw_raw
                    # --- Khâu 1: Publish odom raw (encoder thuần) ---
                    raw_linear_vel = delta_s / max(self.dt, 1e-6)
                    raw_angular_vel = delta_yaw_raw / max(self.dt, 1e-6)
                    self._publish_odom_raw(stamp, delta_s, delta_yaw_raw,
                                          raw_linear_vel, raw_angular_vel)

                    # --- Khâu 2: IMU yaw fusion (tiền xử lý) ---
                    if self.use_imu_for_odom and self.imu_mode != 'none':
                        delta_yaw = (
                            (1.0 - self.imu_yaw_blend) * delta_yaw
                            + self.imu_yaw_blend * self.last_imu_delta_yaw
                        )
                    self.x += delta_s * math.cos(self.yaw + 0.5 * delta_yaw)
                    self.y += delta_s * math.sin(self.yaw + 0.5 * delta_yaw)
                    self.yaw = _normalize_angle(self.yaw + delta_yaw)
                    linear_vel = delta_s / max(self.dt, 1e-6)
                    angular_vel = delta_yaw / max(self.dt, 1e-6)
                    self.current_v = linear_vel
                    self.current_w = angular_vel
                self.prev_left_pos = left
                self.prev_right_pos = right
        elif self.use_open_loop_odom:
            self.x += self.current_v * math.cos(self.yaw) * self.dt
            self.y += self.current_v * math.sin(self.yaw) * self.dt
            self.yaw = _normalize_angle(self.yaw + self.current_w * self.dt)
            linear_vel = self.current_v
            angular_vel = self.current_w

        qx, qy, qz, qw = _quat_from_rpy(0.0, 0.0, self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = linear_vel
        odom.twist.twist.angular.z = angular_vel

        pos_var = 0.03 if self.use_encoder_odom else (0.25 if self.use_open_loop_odom else 1000.0)
        yaw_var = 0.06 if self.use_encoder_odom else (0.50 if self.use_open_loop_odom else 1000.0)
        lin_var = 0.05 if self.use_encoder_odom else (0.30 if self.use_open_loop_odom else 1000.0)
        ang_var = 0.10 if self.use_encoder_odom else (0.50 if self.use_open_loop_odom else 1000.0)

        pose_cov = [0.0] * 36
        twist_cov = [0.0] * 36
        pose_cov[0] = pos_var
        pose_cov[7] = pos_var
        pose_cov[14] = 99999.0
        pose_cov[21] = 99999.0
        pose_cov[28] = 99999.0
        pose_cov[35] = yaw_var
        twist_cov[0] = lin_var
        twist_cov[7] = lin_var
        twist_cov[14] = 99999.0
        twist_cov[21] = 99999.0
        twist_cov[28] = 99999.0
        twist_cov[35] = ang_var
        odom.pose.covariance = pose_cov
        odom.twist.covariance = twist_cov
        self.odom_pub.publish(odom)

        if self.publish_odom_tf:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = self.odom_frame_id
            tf_msg.child_frame_id = self.base_frame_id
            tf_msg.transform.translation.x = self.x
            tf_msg.transform.translation.y = self.y
            tf_msg.transform.translation.z = 0.0
            tf_msg.transform.rotation.x = qx
            tf_msg.transform.rotation.y = qy
            tf_msg.transform.rotation.z = qz
            tf_msg.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf_msg)

    def _publish_odom_raw(self, stamp, delta_s, delta_yaw_raw,
                          linear_vel, angular_vel):
        """Publish encoder-only odometry (Khâu 1: thu dữ liệu)."""
        self.x_raw += delta_s * math.cos(self.yaw_raw + 0.5 * delta_yaw_raw)
        self.y_raw += delta_s * math.sin(self.yaw_raw + 0.5 * delta_yaw_raw)
        self.yaw_raw = _normalize_angle(self.yaw_raw + delta_yaw_raw)

        qx, qy, qz, qw = _quat_from_rpy(0.0, 0.0, self.yaw_raw)

        odom_raw = Odometry()
        odom_raw.header.stamp = stamp
        odom_raw.header.frame_id = self.odom_frame_id
        odom_raw.child_frame_id = self.base_frame_id
        odom_raw.pose.pose.position.x = self.x_raw
        odom_raw.pose.pose.position.y = self.y_raw
        odom_raw.pose.pose.position.z = 0.0
        odom_raw.pose.pose.orientation.x = qx
        odom_raw.pose.pose.orientation.y = qy
        odom_raw.pose.pose.orientation.z = qz
        odom_raw.pose.pose.orientation.w = qw
        odom_raw.twist.twist.linear.x = linear_vel
        odom_raw.twist.twist.angular.z = angular_vel

        raw_cov = [0.0] * 36
        raw_cov[0] = 0.05
        raw_cov[7] = 0.05
        raw_cov[14] = 99999.0
        raw_cov[21] = 99999.0
        raw_cov[28] = 99999.0
        raw_cov[35] = 0.10
        odom_raw.pose.covariance = raw_cov
        odom_raw.twist.covariance = raw_cov
        self.odom_raw_pub.publish(odom_raw)

    def _publish_scan(self, stamp):
        ranges = list(self.lidar.getRangeImage())
        if self.reverse_scan:
            ranges.reverse()

        # Normalize 1 lần duy nhất
        normalized = [self._normalize_scan_range(v) for v in ranges]

        # --- Khâu 1: Publish scan raw (lidar thô, chỉ normalize) ---
        raw_scan = LaserScan()
        raw_scan.header.stamp = stamp
        raw_scan.header.frame_id = self.lidar_frame_id
        raw_scan.angle_min = self.lidar_angle_min
        raw_scan.angle_max = self.lidar_angle_max
        raw_scan.angle_increment = self.lidar_angle_increment
        raw_scan.time_increment = 0.0
        raw_scan.scan_time = self.dt
        raw_scan.range_min = self.lidar_min_range
        raw_scan.range_max = self.lidar_max_range
        raw_scan.ranges = list(normalized)
        raw_scan.intensities = []
        self.scan_raw_pub.publish(raw_scan)

        # --- Khâu 2: Publish scan filtered (tiền xử lý) ---
        clean_ranges = self._filter_scan_ranges(normalized)

        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.lidar_frame_id
        scan.angle_min = self.lidar_angle_min
        scan.angle_max = self.lidar_angle_max
        scan.angle_increment = self.lidar_angle_increment
        scan.time_increment = 0.0
        scan.scan_time = self.dt
        scan.range_min = self.lidar_min_range
        scan.range_max = self.lidar_max_range
        scan.ranges = clean_ranges
        scan.intensities = []
        self.scan_pub.publish(scan)

    def _normalize_scan_range(self, value):
        rng = float(value)
        if not math.isfinite(rng):
            return float('inf')
        if rng < self.lidar_min_range:
            return float('inf')
        if rng > self.lidar_max_range:
            return float('inf')
        if (
            self.scan_use_inf_for_max_range
            and rng >= (self.lidar_max_range - self.scan_max_range_margin)
        ):
            return float('inf')
        return rng

    def _median_filter_scan(self, ranges):
        if self.scan_filter_window <= 1:
            return list(ranges)

        radius = self.scan_filter_window // 2
        filtered = list(ranges)
        total = len(ranges)

        for index, current in enumerate(ranges):
            if not math.isfinite(current):
                continue

            neighbors = []
            for offset in range(-radius, radius + 1):
                sample_index = index + offset
                if sample_index < 0 or sample_index >= total:
                    continue
                sample = ranges[sample_index]
                if math.isfinite(sample):
                    neighbors.append(sample)

            if len(neighbors) < self.scan_filter_min_valid_neighbors:
                continue

            median = _median(neighbors)
            if median is None:
                continue

            if abs(current - median) > self.scan_filter_outlier_threshold_m:
                filtered[index] = median

        return filtered

    def _temporal_filter_scan(self, ranges):
        if (
            self.prev_filtered_ranges is None
            or len(self.prev_filtered_ranges) != len(ranges)
        ):
            self.prev_filtered_ranges = list(ranges)
            return list(ranges)

        alpha = self.scan_filter_temporal_alpha
        jump_threshold = self.scan_filter_temporal_jump_threshold_m
        filtered = list(ranges)

        for index, current in enumerate(ranges):
            previous = self.prev_filtered_ranges[index]
            if not math.isfinite(current):
                filtered[index] = float('inf')
                continue
            if not math.isfinite(previous):
                filtered[index] = current
                continue
            if abs(current - previous) > jump_threshold:
                filtered[index] = current
                continue
            filtered[index] = alpha * current + (1.0 - alpha) * previous

        self.prev_filtered_ranges = list(filtered)
        return filtered

    def _filter_scan_ranges(self, normalized):
        """Filter pre-normalized scan ranges (Khâu 2: tiền xử lý)."""
        if not self.scan_filter_enabled:
            self.prev_filtered_ranges = list(normalized)
            return list(normalized)

        spatially_filtered = self._median_filter_scan(normalized)
        return self._temporal_filter_scan(spatially_filtered)

    def _publish_imu(self, stamp, now_sec):
        if self.imu_mode == 'none':
            return

        if self.last_imu_time_sec is None:
            dt_imu = self.dt
        else:
            dt_imu = now_sec - self.last_imu_time_sec
            if dt_imu <= 0.0 or dt_imu > 0.5:
                dt_imu = self.dt
        self.last_imu_time_sec = now_sec

        raw_msg = Imu()
        raw_msg.header.stamp = stamp
        raw_msg.header.frame_id = self.imu_frame_id
        raw_msg.orientation_covariance[0] = -1.0
        raw_msg.linear_acceleration_covariance[0] = -1.0

        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = self.imu_frame_id

        if self.imu_mode == 'inertial':
            rpy = self._read_rpy(self.imu_inertial)
            if rpy is None:
                return
            roll, pitch, yaw = rpy

            if self.last_imu_rpy is None:
                wx = 0.0
                wy = 0.0
                wz = 0.0
            else:
                wx = _normalize_angle(roll - self.last_imu_rpy[0]) / max(dt_imu, 1e-6)
                wy = _normalize_angle(pitch - self.last_imu_rpy[1]) / max(dt_imu, 1e-6)
                wz = _normalize_angle(yaw - self.last_imu_rpy[2]) / max(dt_imu, 1e-6)
            self.last_imu_rpy = (roll, pitch, yaw)
            self.last_imu_angular_velocity = (wx, wy, wz)
            self.last_imu_delta_yaw = wz * dt_imu

            qx, qy, qz, qw = _quat_from_rpy(roll, pitch, yaw)
            imu_msg.orientation.x = qx
            imu_msg.orientation.y = qy
            imu_msg.orientation.z = qz
            imu_msg.orientation.w = qw
            imu_msg.orientation_covariance[0] = 0.05
            imu_msg.orientation_covariance[4] = 0.05
            imu_msg.orientation_covariance[8] = 0.10
            imu_msg.linear_acceleration_covariance[0] = -1.0
        else:
            gyro = self._read_vec3(self.imu_gyro)
            accel = self._read_vec3(self.imu_accel)
            if gyro is None or accel is None:
                return
            self.last_imu_angular_velocity = gyro
            self.last_imu_linear_accel = accel
            self.last_imu_delta_yaw = self.last_imu_angular_velocity[2] * dt_imu
            imu_msg.orientation_covariance[0] = -1.0
            imu_msg.linear_acceleration.x = accel[0]
            imu_msg.linear_acceleration.y = accel[1]
            imu_msg.linear_acceleration.z = accel[2]
            imu_msg.linear_acceleration_covariance[0] = 0.5
            imu_msg.linear_acceleration_covariance[4] = 0.5
            imu_msg.linear_acceleration_covariance[8] = 0.5

        imu_msg.angular_velocity.x = self.last_imu_angular_velocity[0]
        imu_msg.angular_velocity.y = self.last_imu_angular_velocity[1]
        imu_msg.angular_velocity.z = self.last_imu_angular_velocity[2]
        imu_msg.angular_velocity_covariance[0] = 0.02
        imu_msg.angular_velocity_covariance[4] = 0.02
        imu_msg.angular_velocity_covariance[8] = 0.02

        raw_msg.angular_velocity.x = imu_msg.angular_velocity.x
        raw_msg.angular_velocity.y = imu_msg.angular_velocity.y
        raw_msg.angular_velocity.z = imu_msg.angular_velocity.z
        raw_msg.angular_velocity_covariance = imu_msg.angular_velocity_covariance
        raw_msg.linear_acceleration.x = imu_msg.linear_acceleration.x
        raw_msg.linear_acceleration.y = imu_msg.linear_acceleration.y
        raw_msg.linear_acceleration.z = imu_msg.linear_acceleration.z
        raw_msg.linear_acceleration_covariance = imu_msg.linear_acceleration_covariance

        self.imu_raw_pub.publish(raw_msg)
        self.imu_pub.publish(imu_msg)

    def _next_stamp(self):
        sim_now = float(self.robot.getTime())
        sim_elapsed_ns = int(max(0.0, sim_now - self.stamp_sim_base_sec) * 1e9)
        stamp_ns = self.stamp_wall_base_ns + sim_elapsed_ns
        if stamp_ns <= self.last_stamp_ns:
            stamp_ns = self.last_stamp_ns + 1
        self.last_stamp_ns = stamp_ns

        stamp = TimeMsg()
        stamp.sec = int(stamp_ns // 1_000_000_000)
        stamp.nanosec = int(stamp_ns % 1_000_000_000)
        return stamp

    def step_callback(self):
        if self.robot.step(self.time_step) == -1:
            rclpy.shutdown()
            return

        self._apply_cmd()
        stamp = self._next_stamp()
        now_sec = stamp.sec + (stamp.nanosec * 1e-9)
        self._publish_imu(stamp, now_sec)
        self._publish_odom(stamp)
        # Publish scan after odom/TF so SLAM consumes a consistent transform
        # at the same stamp, especially during in-place rotations.
        self._publish_scan(stamp)


def main(args=None):
    rclpy.init(args=args)
    node = PureWebotsDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
