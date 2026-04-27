import math
from enum import Enum
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class MotionState(Enum):
    ALIGN_TO_TARGET = 'ALIGN_TO_TARGET'
    FORWARD = 'FORWARD'
    AVOID_OBSTACLE = 'AVOID_OBSTACLE'
    BACKUP = 'BACKUP'
    TURN = 'TURN'
    RECOVER_STUCK = 'RECOVER_STUCK'
    GOAL_REACHED = 'GOAL_REACHED'


class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('waypoint_topic', '/coverage_waypoints')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_frame', 'map')

        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('linear_speed', 0.10)
        self.declare_parameter('angular_speed', 0.60)
        self.declare_parameter('min_linear_speed', 0.05)
        self.declare_parameter('max_linear_speed', 0.15)
        self.declare_parameter('max_angular_speed', 0.80)
        self.declare_parameter('backup_speed', 0.06)

        self.declare_parameter('front_stop_distance', 0.35)
        self.declare_parameter('emergency_stop_distance', 0.20)
        self.declare_parameter('align_front_block_distance', -1.0)
        self.declare_parameter('side_clearance', 0.25)
        self.declare_parameter('rear_clearance', 0.22)
        self.declare_parameter('robot_radius', 0.18)
        self.declare_parameter('collision_margin', 0.03)

        self.declare_parameter('backup_duration_sec', 0.80)
        self.declare_parameter('turn_min_duration_sec', 0.75)
        self.declare_parameter('turn_max_duration_sec', 1.60)
        self.declare_parameter('avoid_duration_sec', 1.20)
        self.declare_parameter('avoid_clear_hold_sec', 0.60)
        self.declare_parameter('recover_rotate_duration_sec', 0.90)
        self.declare_parameter('recover_backup_duration_sec', 0.80)
        self.declare_parameter('recover_second_rotate_duration_sec', 1.00)

        self.declare_parameter('state_timeout_sec', 10.0)
        self.declare_parameter('scan_timeout_sec', 1.0)
        self.declare_parameter('odom_timeout_sec', 1.0)
        self.declare_parameter('stuck_timeout_sec', 4.5)
        self.declare_parameter('stuck_min_progress_m', 0.06)

        self.declare_parameter('goal_tolerance_m', 0.20)
        self.declare_parameter('coverage_min_target_distance_m', 0.75)
        self.declare_parameter('align_yaw_tolerance_rad', 0.22)
        self.declare_parameter('slowdown_distance_m', 0.60)
        self.declare_parameter('exploration_lookahead_m', 0.90)
        self.declare_parameter('coverage_skip_timeout_sec', 3.0)
        self.declare_parameter('local_loop_timeout_sec', 24.0)
        self.declare_parameter('local_loop_radius_m', 0.85)
        self.declare_parameter('local_loop_min_path_m', 1.8)
        self.declare_parameter('stop_when_coverage_done', False)
        self.declare_parameter('frontier_enabled', True)
        self.declare_parameter('frontier_search_stride_cells', 1)
        self.declare_parameter('frontier_clearance_cells', 2)
        self.declare_parameter('frontier_min_cluster_size', 10)
        self.declare_parameter('frontier_min_unknown_neighbors', 2)
        self.declare_parameter('frontier_min_distance_m', 0.65)
        self.declare_parameter('frontier_max_distance_m', 12.0)
        self.declare_parameter('frontier_replan_interval_sec', 1.5)
        self.declare_parameter('frontier_target_timeout_sec', 25.0)
        self.declare_parameter('frontier_target_min_hold_sec', 5.0)
        self.declare_parameter('frontier_blocked_skip_timeout_sec', 6.0)
        self.declare_parameter('frontier_blacklist_radius_m', 0.45)
        self.declare_parameter('frontier_blacklist_timeout_sec', 30.0)
        self.declare_parameter('frontier_occupied_threshold', 50)
        self.declare_parameter('frontier_cluster_weight', 0.04)
        self.declare_parameter('frontier_distance_weight', 0.18)
        self.declare_parameter('frontier_heading_weight', 0.30)
        self.declare_parameter('frontier_stickiness_bonus', 0.60)

        self.declare_parameter('planner_horizon_sec', 1.20)
        self.declare_parameter('planner_dt_sec', 0.20)
        self.declare_parameter('planner_linear_samples', 6)
        self.declare_parameter('planner_angular_samples', 11)
        self.declare_parameter('planner_obstacle_max_range', 3.5)
        self.declare_parameter('planner_goal_weight', 2.6)
        self.declare_parameter('planner_clearance_weight', 1.2)
        self.declare_parameter('planner_smoothness_weight', 0.45)
        self.declare_parameter('planner_turn_weight', 0.18)
        self.declare_parameter('planner_heading_weight', 0.50)
        self.declare_parameter('planner_forward_weight', 0.30)
        self.declare_parameter('planner_in_place_penalty', 0.30)

        self.declare_parameter('scan_point_stride', 2)
        self.declare_parameter('sector_percentile', 0.25)
        self.declare_parameter('tf_lookup_timeout_sec', 0.15)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.waypoint_topic = str(self.get_parameter('waypoint_topic').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)

        self.control_rate_hz = max(2.0, float(self.get_parameter('control_rate_hz').value))
        self.linear_speed = max(0.02, float(self.get_parameter('linear_speed').value))
        self.angular_speed = max(0.10, float(self.get_parameter('angular_speed').value))
        self.min_linear_speed = max(0.01, float(self.get_parameter('min_linear_speed').value))
        self.max_linear_speed = max(0.02, float(self.get_parameter('max_linear_speed').value))
        self.max_angular_speed = max(0.10, float(self.get_parameter('max_angular_speed').value))
        self.backup_speed = max(0.02, float(self.get_parameter('backup_speed').value))

        self.linear_speed = clamp(self.linear_speed, self.min_linear_speed, self.max_linear_speed)
        self.angular_speed = min(self.angular_speed, self.max_angular_speed)

        self.front_stop_distance = max(
            0.10, float(self.get_parameter('front_stop_distance').value)
        )
        self.emergency_stop_distance = max(
            0.05, float(self.get_parameter('emergency_stop_distance').value)
        )
        self.align_front_block_distance = float(
            self.get_parameter('align_front_block_distance').value
        )
        if self.align_front_block_distance <= 0.0:
            self.align_front_block_distance = max(
                self.emergency_stop_distance + 0.03,
                self.front_stop_distance - 0.08,
            )
        else:
            self.align_front_block_distance = clamp(
                self.align_front_block_distance,
                self.emergency_stop_distance,
                self.front_stop_distance,
            )
        self.side_clearance = max(0.05, float(self.get_parameter('side_clearance').value))
        self.rear_clearance = max(0.05, float(self.get_parameter('rear_clearance').value))
        self.robot_radius = max(0.05, float(self.get_parameter('robot_radius').value))
        self.collision_margin = max(0.0, float(self.get_parameter('collision_margin').value))

        self.backup_duration_sec = max(
            0.2, float(self.get_parameter('backup_duration_sec').value)
        )
        self.turn_min_duration_sec = max(
            0.2, float(self.get_parameter('turn_min_duration_sec').value)
        )
        self.turn_max_duration_sec = max(
            self.turn_min_duration_sec,
            float(self.get_parameter('turn_max_duration_sec').value),
        )
        self.avoid_duration_sec = max(
            0.3, float(self.get_parameter('avoid_duration_sec').value)
        )
        self.avoid_clear_hold_sec = max(
            0.1, float(self.get_parameter('avoid_clear_hold_sec').value)
        )
        self.recover_rotate_duration_sec = max(
            0.3, float(self.get_parameter('recover_rotate_duration_sec').value)
        )
        self.recover_backup_duration_sec = max(
            0.3, float(self.get_parameter('recover_backup_duration_sec').value)
        )
        self.recover_second_rotate_duration_sec = max(
            0.3, float(self.get_parameter('recover_second_rotate_duration_sec').value)
        )

        self.state_timeout_sec = max(1.0, float(self.get_parameter('state_timeout_sec').value))
        self.scan_timeout_sec = max(0.2, float(self.get_parameter('scan_timeout_sec').value))
        self.odom_timeout_sec = max(0.2, float(self.get_parameter('odom_timeout_sec').value))
        self.stuck_timeout_sec = max(1.0, float(self.get_parameter('stuck_timeout_sec').value))
        self.stuck_min_progress_m = max(
            0.01, float(self.get_parameter('stuck_min_progress_m').value)
        )

        self.goal_tolerance_m = max(0.05, float(self.get_parameter('goal_tolerance_m').value))
        self.coverage_min_target_distance_m = max(
            self.goal_tolerance_m,
            float(self.get_parameter('coverage_min_target_distance_m').value),
        )
        self.align_yaw_tolerance_rad = max(
            0.05, float(self.get_parameter('align_yaw_tolerance_rad').value)
        )
        self.slowdown_distance_m = max(0.15, float(self.get_parameter('slowdown_distance_m').value))
        self.exploration_lookahead_m = max(
            0.2, float(self.get_parameter('exploration_lookahead_m').value)
        )
        self.coverage_skip_timeout_sec = max(
            0.5, float(self.get_parameter('coverage_skip_timeout_sec').value)
        )
        self.local_loop_timeout_sec = max(
            5.0, float(self.get_parameter('local_loop_timeout_sec').value)
        )
        self.local_loop_radius_m = max(
            0.2, float(self.get_parameter('local_loop_radius_m').value)
        )
        self.local_loop_min_path_m = max(
            0.2, float(self.get_parameter('local_loop_min_path_m').value)
        )
        self.stop_when_coverage_done = bool(
            self.get_parameter('stop_when_coverage_done').value
        )
        self.frontier_enabled = bool(self.get_parameter('frontier_enabled').value)
        self.frontier_search_stride_cells = max(
            1, int(self.get_parameter('frontier_search_stride_cells').value)
        )
        self.frontier_clearance_cells = max(
            1, int(self.get_parameter('frontier_clearance_cells').value)
        )
        self.frontier_min_cluster_size = max(
            1, int(self.get_parameter('frontier_min_cluster_size').value)
        )
        self.frontier_min_unknown_neighbors = max(
            1, int(self.get_parameter('frontier_min_unknown_neighbors').value)
        )
        self.frontier_min_distance_m = max(
            0.10, float(self.get_parameter('frontier_min_distance_m').value)
        )
        self.frontier_max_distance_m = max(
            self.frontier_min_distance_m,
            float(self.get_parameter('frontier_max_distance_m').value),
        )
        self.frontier_replan_interval_sec = max(
            0.2, float(self.get_parameter('frontier_replan_interval_sec').value)
        )
        self.frontier_target_timeout_sec = max(
            2.0, float(self.get_parameter('frontier_target_timeout_sec').value)
        )
        self.frontier_target_min_hold_sec = max(
            0.0, float(self.get_parameter('frontier_target_min_hold_sec').value)
        )
        self.frontier_blocked_skip_timeout_sec = max(
            1.0, float(self.get_parameter('frontier_blocked_skip_timeout_sec').value)
        )
        self.frontier_blacklist_radius_m = max(
            0.05, float(self.get_parameter('frontier_blacklist_radius_m').value)
        )
        self.frontier_blacklist_timeout_sec = max(
            1.0, float(self.get_parameter('frontier_blacklist_timeout_sec').value)
        )
        self.frontier_occupied_threshold = int(
            self.get_parameter('frontier_occupied_threshold').value
        )
        self.frontier_cluster_weight = float(
            self.get_parameter('frontier_cluster_weight').value
        )
        self.frontier_distance_weight = float(
            self.get_parameter('frontier_distance_weight').value
        )
        self.frontier_heading_weight = float(
            self.get_parameter('frontier_heading_weight').value
        )
        self.frontier_stickiness_bonus = float(
            self.get_parameter('frontier_stickiness_bonus').value
        )

        self.planner_horizon_sec = max(
            0.4, float(self.get_parameter('planner_horizon_sec').value)
        )
        self.planner_dt_sec = max(0.05, float(self.get_parameter('planner_dt_sec').value))
        self.planner_linear_samples = max(
            3, int(self.get_parameter('planner_linear_samples').value)
        )
        self.planner_angular_samples = max(
            3, int(self.get_parameter('planner_angular_samples').value)
        )
        self.planner_obstacle_max_range = max(
            0.5, float(self.get_parameter('planner_obstacle_max_range').value)
        )
        self.planner_goal_weight = float(self.get_parameter('planner_goal_weight').value)
        self.planner_clearance_weight = float(
            self.get_parameter('planner_clearance_weight').value
        )
        self.planner_smoothness_weight = float(
            self.get_parameter('planner_smoothness_weight').value
        )
        self.planner_turn_weight = float(self.get_parameter('planner_turn_weight').value)
        self.planner_heading_weight = float(self.get_parameter('planner_heading_weight').value)
        self.planner_forward_weight = float(
            self.get_parameter('planner_forward_weight').value
        )
        self.planner_in_place_penalty = float(
            self.get_parameter('planner_in_place_penalty').value
        )

        self.scan_point_stride = max(1, int(self.get_parameter('scan_point_stride').value))
        self.sector_percentile = clamp(
            float(self.get_parameter('sector_percentile').value), 0.01, 0.95
        )
        self.tf_lookup_timeout_sec = max(
            0.01, float(self.get_parameter('tf_lookup_timeout_sec').value)
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)
        self.waypoint_sub = self.create_subscription(
            PoseArray, self.waypoint_topic, self.waypoint_cb, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        timer_period = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.state = MotionState.ALIGN_TO_TARGET
        self.state_enter_time = self.now_sec()
        self.backup_until = self.state_enter_time
        self.turn_until = self.state_enter_time
        self.avoid_until = self.state_enter_time
        self.turn_sign = 1.0
        self.last_turn_sign = 1.0

        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.sector_min = {
            'front': float('inf'),
            'left': float('inf'),
            'right': float('inf'),
            'front_left': float('inf'),
            'front_right': float('inf'),
            'rear': float('inf'),
        }
        self.sector_p = {
            'front': float('inf'),
            'left': float('inf'),
            'right': float('inf'),
            'front_left': float('inf'),
            'front_right': float('inf'),
            'rear': float('inf'),
        }
        self.scan_points: List[Tuple[float, float]] = []
        self.scan_range_max = 8.0
        self.map_msg: Optional[OccupancyGrid] = None
        self.last_map_time = 0.0

        self.have_pose = False
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.last_progress_time = self.now_sec()
        self.progress_ref_x = 0.0
        self.progress_ref_y = 0.0
        self.loop_anchor_x = 0.0
        self.loop_anchor_y = 0.0
        self.loop_anchor_time = self.now_sec()
        self.loop_path_accum = 0.0
        self.loop_last_x = 0.0
        self.loop_last_y = 0.0
        self.loop_have_last_pose = False

        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0

        self.recover_phase = 0
        self.recover_phase_until = self.state_enter_time
        self.recover_turn_sign = 1.0

        self.waypoint_frame = self.odom_frame
        self.waypoints: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.last_waypoint_count = 0
        self.last_waypoint_signature = ()
        self.last_coverage_target_odom: Optional[Tuple[float, float]] = None
        self.coverage_blocked_since: Optional[float] = None
        self.avoid_clear_since: Optional[float] = None
        self.last_tf_warn_time = 0.0
        self.frontier_target_map: Optional[Tuple[float, float]] = None
        self.frontier_target_odom: Optional[Tuple[float, float]] = None
        self.frontier_target_selected_time = 0.0
        self.frontier_last_replan_time = 0.0
        self.frontier_distance_best = float('inf')
        self.frontier_blacklist: List[Tuple[float, float, float]] = []
        self.frontier_blocked_since: Optional[float] = None
        self.last_target_source = ''

        self.get_logger().info(
            'Smart motion controller started: '
            f'scan={self.scan_topic}, map={self.map_topic}, odom={self.odom_topic}, '
            f'cmd_vel={self.cmd_vel_topic}, waypoints={self.waypoint_topic}'
        )

    def now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _clean_scan_value(self, msg: LaserScan, value: float) -> Optional[float]:
        val = float(value)
        if math.isnan(val):
            return None
        if math.isinf(val):
            val = float(msg.range_max)
        if val < float(msg.range_min):
            return None
        max_range = float(msg.range_max) if math.isfinite(msg.range_max) else self.scan_range_max
        if val > max_range:
            val = max_range
        return val

    def _sector_values(self, msg: LaserScan, start_deg: float, end_deg: float) -> List[float]:
        values: List[float] = []
        wrapped = start_deg > end_deg
        angle = float(msg.angle_min)
        angle_inc = float(msg.angle_increment)
        if angle_inc == 0.0:
            return values

        for raw in msg.ranges:
            clean = self._clean_scan_value(msg, raw)
            deg = math.degrees(normalize_angle(angle))
            angle += angle_inc
            if clean is None:
                continue
            if wrapped:
                if deg >= start_deg or deg <= end_deg:
                    values.append(clean)
            else:
                if start_deg <= deg <= end_deg:
                    values.append(clean)
        return values

    def _sector_stats(self, msg: LaserScan, start_deg: float, end_deg: float) -> Tuple[float, float]:
        values = self._sector_values(msg, start_deg, end_deg)
        fallback = float(msg.range_max) if math.isfinite(msg.range_max) else self.scan_range_max
        if not values:
            return fallback, fallback

        values.sort()
        idx = int(self.sector_percentile * (len(values) - 1))
        percentile = values[idx]
        return values[0], percentile

    def scan_cb(self, msg):
        self.last_scan_time = self.now_sec()
        self.scan_range_max = float(msg.range_max) if math.isfinite(msg.range_max) else 8.0

        front_min, front_p = self._sector_stats(msg, -20.0, 20.0)
        fl_min, fl_p = self._sector_stats(msg, 20.0, 70.0)
        left_min, left_p = self._sector_stats(msg, 70.0, 120.0)
        fr_min, fr_p = self._sector_stats(msg, -70.0, -20.0)
        right_min, right_p = self._sector_stats(msg, -120.0, -70.0)
        rear_min, rear_p = self._sector_stats(msg, 150.0, -150.0)

        self.sector_min['front'] = front_min
        self.sector_min['front_left'] = fl_min
        self.sector_min['left'] = left_min
        self.sector_min['front_right'] = fr_min
        self.sector_min['right'] = right_min
        self.sector_min['rear'] = rear_min

        self.sector_p['front'] = front_p
        self.sector_p['front_left'] = fl_p
        self.sector_p['left'] = left_p
        self.sector_p['front_right'] = fr_p
        self.sector_p['right'] = right_p
        self.sector_p['rear'] = rear_p

        self.scan_points = []
        angle = float(msg.angle_min)
        max_range = min(self.scan_range_max, self.planner_obstacle_max_range)
        for idx, raw in enumerate(msg.ranges):
            clean = self._clean_scan_value(msg, raw)
            if clean is not None and clean <= max_range and (idx % self.scan_point_stride) == 0:
                self.scan_points.append((clean * math.cos(angle), clean * math.sin(angle)))
            angle += float(msg.angle_increment)

    def odom_cb(self, msg):
        self.pose_x = float(msg.pose.pose.position.x)
        self.pose_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.pose_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.last_odom_time = self.now_sec()

        if not self.have_pose:
            self.have_pose = True
            self.progress_ref_x = self.pose_x
            self.progress_ref_y = self.pose_y
            self.last_progress_time = self.now_sec()
            self.loop_anchor_x = self.pose_x
            self.loop_anchor_y = self.pose_y
            self.loop_anchor_time = self.now_sec()
            self.loop_path_accum = 0.0
            self.loop_last_x = self.pose_x
            self.loop_last_y = self.pose_y
            self.loop_have_last_pose = True
            return

        if self.loop_have_last_pose:
            self.loop_path_accum += math.hypot(
                self.pose_x - self.loop_last_x,
                self.pose_y - self.loop_last_y,
            )
        self.loop_last_x = self.pose_x
        self.loop_last_y = self.pose_y
        self.loop_have_last_pose = True

    def map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.last_map_time = self.now_sec()
        frame_id = str(msg.header.frame_id).strip()
        if frame_id:
            self.map_frame = frame_id

    def waypoint_cb(self, msg: PoseArray):
        frame_id = str(msg.header.frame_id).strip()
        waypoint_frame = frame_id if frame_id else self.odom_frame
        waypoints = [
            (float(pose.position.x), float(pose.position.y))
            for pose in msg.poses
        ]
        signature = tuple((round(x, 3), round(y, 3)) for x, y in waypoints)
        changed = (
            waypoint_frame != self.waypoint_frame
            or signature != self.last_waypoint_signature
        )

        self.waypoint_frame = waypoint_frame
        self.waypoints = waypoints
        self.last_waypoint_signature = signature

        if changed:
            self.current_waypoint_idx = 0
            self.coverage_blocked_since = None
            self._clear_frontier_target()
            reused_previous_coverage_target = False
            if self.last_coverage_target_odom is not None and self.waypoints:
                best_idx = None
                best_dist = float('inf')
                for idx, (wx, wy) in enumerate(self.waypoints):
                    transformed = self._transform_point_to_odom(
                        wx, wy, self.waypoint_frame
                    )
                    if transformed is None:
                        break
                    distance = math.hypot(
                        transformed[0] - self.last_coverage_target_odom[0],
                        transformed[1] - self.last_coverage_target_odom[1],
                    )
                    if distance < best_dist:
                        best_dist = distance
                        best_idx = idx

                if best_idx is not None:
                    self.current_waypoint_idx = best_idx
                    reused_previous_coverage_target = True

            if (not reused_previous_coverage_target) and self.have_pose and self.waypoints:
                min_target_distance = max(
                    1.25 * self.goal_tolerance_m,
                    self.coverage_min_target_distance_m,
                )
                for idx, (wx, wy) in enumerate(self.waypoints):
                    transformed = self._transform_point_to_odom(
                        wx, wy, self.waypoint_frame
                    )
                    if transformed is None:
                        break
                    distance = math.hypot(
                        transformed[0] - self.pose_x,
                        transformed[1] - self.pose_y,
                    )
                    if distance >= min_target_distance:
                        self.current_waypoint_idx = idx
                        break
        elif self.current_waypoint_idx >= len(self.waypoints):
            self.current_waypoint_idx = 0

        if changed or len(self.waypoints) != self.last_waypoint_count:
            self.get_logger().info(
                f'Received {len(self.waypoints)} coverage waypoint(s) in frame {self.waypoint_frame}'
            )
            self.last_waypoint_count = len(self.waypoints)

    def _transform_point_to_odom(
        self, x: float, y: float, source_frame: str
    ) -> Optional[Tuple[float, float]]:
        if source_frame == self.odom_frame:
            return x, y

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.odom_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
        except TransformException:
            now = self.now_sec()
            if (now - self.last_tf_warn_time) > 2.0:
                self.get_logger().warn(
                    f'Cannot transform waypoint frame {source_frame} -> {self.odom_frame} yet.'
                )
                self.last_tf_warn_time = now
            return None

        tx = float(tf_msg.transform.translation.x)
        ty = float(tf_msg.transform.translation.y)
        q = tf_msg.transform.rotation
        tyaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        c = math.cos(tyaw)
        s = math.sin(tyaw)
        return tx + (c * x) - (s * y), ty + (s * x) + (c * y)

    def _robot_pose_in_frame(
        self, target_frame: str
    ) -> Optional[Tuple[float, float, float]]:
        if not self.have_pose:
            return None

        if target_frame == self.odom_frame:
            return self.pose_x, self.pose_y, self.pose_yaw

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                self.odom_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
        except TransformException:
            return None

        tx = float(tf_msg.transform.translation.x)
        ty = float(tf_msg.transform.translation.y)
        q = tf_msg.transform.rotation
        tyaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        c = math.cos(tyaw)
        s = math.sin(tyaw)
        x = tx + (c * self.pose_x) - (s * self.pose_y)
        y = ty + (s * self.pose_x) + (c * self.pose_y)
        yaw = normalize_angle(self.pose_yaw + tyaw)
        return x, y, yaw

    def _choose_turn_direction(self):
        left_open = min(self.sector_p['left'], self.sector_p['front_left'])
        right_open = min(self.sector_p['right'], self.sector_p['front_right'])

        if abs(left_open - right_open) < 0.05:
            self.turn_sign = -self.last_turn_sign
        elif left_open > right_open:
            self.turn_sign = 1.0
        else:
            self.turn_sign = -1.0

        self.last_turn_sign = self.turn_sign
        return self.turn_sign

    def _transition(self, new_state: MotionState, reason: str):
        if self.state == new_state:
            return

        now = self.now_sec()
        self.state = new_state
        self.state_enter_time = now
        self.get_logger().info(f'State -> {self.state.value}: {reason}')

        if self.state == MotionState.BACKUP:
            self.backup_until = now + self.backup_duration_sec
            self._choose_turn_direction()
        elif self.state == MotionState.TURN:
            self._choose_turn_direction()
            front_clearance = self.sector_min['front']
            if front_clearance < self.front_stop_distance:
                turn_duration = self.turn_max_duration_sec
            elif front_clearance < (self.front_stop_distance + 0.20):
                turn_duration = 0.5 * (
                    self.turn_min_duration_sec + self.turn_max_duration_sec
                )
            else:
                turn_duration = self.turn_min_duration_sec
            self.turn_until = now + turn_duration
        elif self.state == MotionState.AVOID_OBSTACLE:
            self.avoid_until = now + self.avoid_duration_sec
            self.avoid_clear_since = None
            self._choose_turn_direction()
        elif self.state == MotionState.RECOVER_STUCK:
            self.recover_phase = 0
            self.recover_turn_sign = self._choose_turn_direction()
            self.recover_phase_until = now + self.recover_rotate_duration_sec

    def publish_cmd(self, linear, angular):
        if not rclpy.ok():
            return
        linear = clamp(float(linear), -self.max_linear_speed, self.max_linear_speed)
        angular = clamp(float(angular), -self.max_angular_speed, self.max_angular_speed)
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        try:
            self.cmd_pub.publish(msg)
            self.last_cmd_linear = linear
            self.last_cmd_angular = angular
        except Exception:
            return

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def _update_progress_tracker(self, now):
        if not self.have_pose:
            return
        traveled = math.hypot(
            self.pose_x - self.progress_ref_x,
            self.pose_y - self.progress_ref_y,
        )
        if traveled >= self.stuck_min_progress_m:
            self.progress_ref_x = self.pose_x
            self.progress_ref_y = self.pose_y
            self.last_progress_time = now

    def _check_stuck(self, now):
        if not self.have_pose:
            return False

        self._update_progress_tracker(now)

        if self.state not in (
            MotionState.FORWARD,
            MotionState.ALIGN_TO_TARGET,
            MotionState.AVOID_OBSTACLE,
        ):
            return False

        if abs(self.last_cmd_linear) < 0.03:
            self.last_progress_time = now
            return False

        return (now - self.last_progress_time) > self.stuck_timeout_sec

    def _state_timed_out(self, now: float) -> bool:
        elapsed = now - self.state_enter_time
        if self.state in (
            MotionState.BACKUP,
            MotionState.TURN,
            MotionState.AVOID_OBSTACLE,
            MotionState.RECOVER_STUCK,
        ):
            return elapsed > self.state_timeout_sec
        if self.state == MotionState.ALIGN_TO_TARGET:
            return elapsed > min(self.state_timeout_sec, 6.0)
        return False

    def _map_cell_value(self, data, width: int, height: int, x: int, y: int) -> int:
        if x < 0 or y < 0 or x >= width or y >= height:
            return self.frontier_occupied_threshold
        return int(data[y * width + x])

    def _is_frontier_cell(self, data, width: int, height: int, x: int, y: int) -> bool:
        value = self._map_cell_value(data, width, height, x, y)
        if value != 0:
            return False

        unknown_neighbors = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = self._map_cell_value(data, width, height, x + dx, y + dy)
                if neighbor < 0:
                    unknown_neighbors += 1

        if unknown_neighbors < self.frontier_min_unknown_neighbors:
            return False

        radius = self.frontier_clearance_cells
        radius_sq = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dx * dx + dy * dy) > radius_sq:
                    continue
                neighbor = self._map_cell_value(data, width, height, x + dx, y + dy)
                if neighbor >= self.frontier_occupied_threshold:
                    return False

        return True

    def _extract_frontier_clusters(self) -> List[Tuple[float, float, int]]:
        if self.map_msg is None:
            return []

        info = self.map_msg.info
        width = int(info.width)
        height = int(info.height)
        if width <= 0 or height <= 0:
            return []

        data = self.map_msg.data
        step = self.frontier_search_stride_cells
        start = self.frontier_clearance_cells
        frontier_cells = set()

        for y in range(start, max(start, height - start), step):
            for x in range(start, max(start, width - start), step):
                if self._is_frontier_cell(data, width, height, x, y):
                    frontier_cells.add((x, y))

        if not frontier_cells:
            return []

        neighbor_offsets = []
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dx == 0 and dy == 0:
                    continue
                neighbor_offsets.append((dx, dy))

        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        resolution = float(info.resolution)
        clusters: List[Tuple[float, float, int]] = []

        while frontier_cells:
            seed = frontier_cells.pop()
            stack = [seed]
            cluster = [seed]

            while stack:
                cx, cy = stack.pop()
                for dx, dy in neighbor_offsets:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in frontier_cells:
                        frontier_cells.remove(neighbor)
                        stack.append(neighbor)
                        cluster.append(neighbor)

            if len(cluster) < self.frontier_min_cluster_size:
                continue

            mean_x = sum(cell[0] for cell in cluster) / float(len(cluster))
            mean_y = sum(cell[1] for cell in cluster) / float(len(cluster))
            best_cell = min(
                cluster,
                key=lambda cell: ((cell[0] - mean_x) ** 2) + ((cell[1] - mean_y) ** 2),
            )

            wx = origin_x + (best_cell[0] + 0.5) * resolution
            wy = origin_y + (best_cell[1] + 0.5) * resolution
            clusters.append((wx, wy, len(cluster)))

        return clusters

    def _prune_frontier_blacklist(self, now: float):
        self.frontier_blacklist = [
            item for item in self.frontier_blacklist if item[2] > now
        ]

    def _is_frontier_blacklisted(self, x: float, y: float) -> bool:
        for bx, by, _ in self.frontier_blacklist:
            if math.hypot(x - bx, y - by) <= self.frontier_blacklist_radius_m:
                return True
        return False

    def _clear_frontier_target(self):
        self.frontier_target_map = None
        self.frontier_target_odom = None
        self.frontier_target_selected_time = 0.0
        self.frontier_distance_best = float('inf')
        self.frontier_blocked_since = None

    def _blacklist_current_frontier_target(self, reason: str):
        if self.frontier_target_map is not None:
            expiry = self.now_sec() + self.frontier_blacklist_timeout_sec
            self.frontier_blacklist.append(
                (self.frontier_target_map[0], self.frontier_target_map[1], expiry)
            )
            self.get_logger().warn(f'Frontier target blacklisted: {reason}')
        self._clear_frontier_target()

    def _select_frontier_target(self) -> Optional[Tuple[float, float]]:
        if not self.frontier_enabled or self.map_msg is None or not self.have_pose:
            return None

        now = self.now_sec()
        self._prune_frontier_blacklist(now)

        if self.frontier_target_odom is not None:
            distance = math.hypot(
                self.frontier_target_odom[0] - self.pose_x,
                self.frontier_target_odom[1] - self.pose_y,
            )
            self.frontier_distance_best = min(self.frontier_distance_best, distance)
            if distance <= self.goal_tolerance_m:
                self.get_logger().info('Frontier target reached. Replanning.')
                self._clear_frontier_target()
            elif (now - self.frontier_target_selected_time) > self.frontier_target_timeout_sec:
                self._blacklist_current_frontier_target('frontier target timeout')
            elif (now - self.frontier_target_selected_time) < self.frontier_target_min_hold_sec:
                return self.frontier_target_odom
            elif (now - self.frontier_last_replan_time) < self.frontier_replan_interval_sec:
                return self.frontier_target_odom

        robot_pose = self._robot_pose_in_frame(self.map_frame)
        if robot_pose is None:
            return self.frontier_target_odom

        robot_x, robot_y, robot_yaw = robot_pose
        clusters = self._extract_frontier_clusters()
        self.frontier_last_replan_time = now

        best_score = -1e9
        best_map_target = None
        best_odom_target = None
        best_cluster_size = 0

        for wx, wy, cluster_size in clusters:
            if self._is_frontier_blacklisted(wx, wy):
                continue

            dx = wx - robot_x
            dy = wy - robot_y
            distance = math.hypot(dx, dy)
            if distance < self.frontier_min_distance_m or distance > self.frontier_max_distance_m:
                continue

            yaw_error = abs(normalize_angle(math.atan2(dy, dx) - robot_yaw))
            odom_target = self._transform_point_to_odom(wx, wy, self.map_frame)
            if odom_target is None:
                continue

            score = (
                self.frontier_cluster_weight * float(cluster_size)
                - self.frontier_distance_weight * distance
                - self.frontier_heading_weight * yaw_error
            )

            if self.frontier_target_map is not None:
                sticky_distance = math.hypot(
                    wx - self.frontier_target_map[0],
                    wy - self.frontier_target_map[1],
                )
                if sticky_distance <= max(
                    self.frontier_blacklist_radius_m, 2.0 * self.goal_tolerance_m
                ):
                    score += self.frontier_stickiness_bonus

            if score > best_score:
                best_score = score
                best_map_target = (wx, wy)
                best_odom_target = odom_target
                best_cluster_size = cluster_size

        if best_map_target is None or best_odom_target is None:
            return self.frontier_target_odom

        target_changed = (
            self.frontier_target_map is None
            or math.hypot(
                best_map_target[0] - self.frontier_target_map[0],
                best_map_target[1] - self.frontier_target_map[1],
            )
            > self.goal_tolerance_m
        )
        self.frontier_target_map = best_map_target
        self.frontier_target_odom = best_odom_target
        if target_changed:
            self.frontier_target_selected_time = now
            self.frontier_distance_best = math.hypot(
                best_odom_target[0] - self.pose_x,
                best_odom_target[1] - self.pose_y,
            )
            self.get_logger().info(
                'Frontier target selected: '
                f'cluster_size={best_cluster_size}, '
                f'distance={self.frontier_distance_best:.2f}m'
            )
        return self.frontier_target_odom

    def _exploration_target(self) -> Tuple[float, float]:
        left_open = min(self.sector_p['left'], self.sector_p['front_left'])
        right_open = min(self.sector_p['right'], self.sector_p['front_right'])
        front_open = self.sector_p['front']

        bias = clamp(0.9 * (left_open - right_open), -0.9, 0.9)
        if front_open < (self.front_stop_distance + 0.10):
            bias = 0.9 if left_open >= right_open else -0.9

        lookahead = clamp(0.75 * front_open, 0.45, self.exploration_lookahead_m)
        target_heading = normalize_angle(self.pose_yaw + bias)
        return (
            self.pose_x + lookahead * math.cos(target_heading),
            self.pose_y + lookahead * math.sin(target_heading),
        )

    def _resolve_target(self) -> Tuple[Optional[Tuple[float, float]], str]:
        if self.waypoints:
            while self.current_waypoint_idx < len(self.waypoints):
                wx, wy = self.waypoints[self.current_waypoint_idx]
                transformed = self._transform_point_to_odom(wx, wy, self.waypoint_frame)
                if transformed is None:
                    break

                gx, gy = transformed
                distance = math.hypot(gx - self.pose_x, gy - self.pose_y)
                if distance <= self.goal_tolerance_m:
                    self.get_logger().info(
                        f'Coverage waypoint {self.current_waypoint_idx} reached.'
                    )
                    self.current_waypoint_idx += 1
                    self.coverage_blocked_since = None
                    continue

                if (
                    distance < max(1.25 * self.goal_tolerance_m, self.coverage_min_target_distance_m)
                    and (self.current_waypoint_idx + 1) < len(self.waypoints)
                ):
                    self.current_waypoint_idx += 1
                    continue

                self.last_coverage_target_odom = (gx, gy)
                return (gx, gy), 'coverage'

            if self.current_waypoint_idx >= len(self.waypoints):
                if self.stop_when_coverage_done:
                    return None, 'coverage_done'
                frontier_target = self._select_frontier_target()
                if frontier_target is not None:
                    return frontier_target, 'frontier'
                return self._exploration_target(), 'explore'

        frontier_target = self._select_frontier_target()
        if frontier_target is not None:
            return frontier_target, 'frontier'
        return self._exploration_target(), 'explore'

    def _goal_metrics(self, target: Tuple[float, float]) -> Tuple[float, float]:
        tx, ty = target
        dx = tx - self.pose_x
        dy = ty - self.pose_y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(target_yaw - self.pose_yaw)
        return distance, yaw_error

    def _goal_relative_to_robot(self, target: Tuple[float, float]) -> Tuple[float, float]:
        tx, ty = target
        dx = tx - self.pose_x
        dy = ty - self.pose_y
        c = math.cos(self.pose_yaw)
        s = math.sin(self.pose_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _trajectory_clearance(self, x: float, y: float) -> float:
        if not self.scan_points:
            return self.planner_obstacle_max_range
        min_dist = self.planner_obstacle_max_range
        for ox, oy in self.scan_points:
            dist = math.hypot(ox - x, oy - y)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _sample_velocity_command(
        self, target: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        goal_rx, goal_ry = self._goal_relative_to_robot(target)
        start_dist = math.hypot(goal_rx, goal_ry)
        if start_dist <= self.goal_tolerance_m:
            return 0.0, 0.0

        horizon = self.planner_horizon_sec
        dt = self.planner_dt_sec
        steps = max(2, int(horizon / dt))

        max_v = min(self.linear_speed, self.max_linear_speed)
        if self.sector_p['front'] < (self.front_stop_distance + 0.20):
            max_v = min(max_v, 0.08)

        linear_candidates = [0.0]
        if max_v > 0.0:
            linear_candidates.append(min(0.03, max_v))
        if self.planner_linear_samples == 1:
            linear_candidates.append(max_v)
        else:
            for i in range(self.planner_linear_samples):
                ratio = float(i) / float(self.planner_linear_samples - 1)
                linear_candidates.append(self.min_linear_speed + ratio * (max_v - self.min_linear_speed))
        linear_candidates = sorted(set(round(v, 4) for v in linear_candidates))

        angular_candidates = []
        if self.planner_angular_samples == 1:
            angular_candidates = [0.0]
        else:
            for i in range(self.planner_angular_samples):
                ratio = float(i) / float(self.planner_angular_samples - 1)
                angular_candidates.append(
                    -self.max_angular_speed + (2.0 * self.max_angular_speed * ratio)
                )

        best_score = -1e9
        best_cmd: Optional[Tuple[float, float]] = None
        safety_radius = self.robot_radius + self.collision_margin
        if (
            start_dist > (2.0 * self.goal_tolerance_m)
            and self.sector_p['front'] > (self.front_stop_distance + 0.08)
        ):
            safety_radius = max(0.6 * self.robot_radius, 0.92 * safety_radius)

        for v in linear_candidates:
            for w in angular_candidates:
                if abs(v) < 1e-5 and abs(w) < 0.12:
                    continue

                x = 0.0
                y = 0.0
                yaw = 0.0
                min_clearance = self.planner_obstacle_max_range
                collision = False

                for _ in range(steps):
                    x += v * math.cos(yaw) * dt
                    y += v * math.sin(yaw) * dt
                    yaw = normalize_angle(yaw + w * dt)

                    clearance = self._trajectory_clearance(x, y)
                    min_clearance = min(min_clearance, clearance)
                    if clearance < safety_radius:
                        collision = True
                        break

                if collision:
                    continue

                end_dist = math.hypot(goal_rx - x, goal_ry - y)
                progress = start_dist - end_dist
                goal_heading = math.atan2(goal_ry - y, goal_rx - x)
                heading_err = abs(normalize_angle(goal_heading - yaw))

                smooth_penalty = abs(v - self.last_cmd_linear) + 0.5 * abs(w - self.last_cmd_angular)
                clearance_score = clamp(min_clearance - safety_radius, 0.0, 2.0)

                score = (
                    self.planner_goal_weight * progress
                    + self.planner_clearance_weight * clearance_score
                    - self.planner_smoothness_weight * smooth_penalty
                    - self.planner_turn_weight * abs(w)
                    - self.planner_heading_weight * heading_err
                    + self.planner_forward_weight * v
                )

                if (
                    start_dist > (2.0 * self.goal_tolerance_m)
                    and self.sector_p['front'] > (self.front_stop_distance + 0.12)
                    and v < (0.9 * self.min_linear_speed)
                ):
                    score -= self.planner_in_place_penalty

                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)

        return best_cmd

    def _run_recover(self, now: float):
        if self.recover_phase == 0:
            if now < self.recover_phase_until:
                self.publish_cmd(0.0, self.recover_turn_sign * 0.7 * self.angular_speed)
                return
            self.recover_phase = 1
            self.recover_phase_until = now + self.recover_backup_duration_sec

        if self.recover_phase == 1:
            if now < self.recover_phase_until:
                self.publish_cmd(-self.backup_speed, -0.2 * self.recover_turn_sign)
                return
            self.recover_phase = 2
            self.recover_phase_until = now + self.recover_second_rotate_duration_sec

        if self.recover_phase == 2:
            if now < self.recover_phase_until:
                self.publish_cmd(0.0, -self.recover_turn_sign * 0.8 * self.angular_speed)
                return
            self._transition(MotionState.ALIGN_TO_TARGET, 'recover sequence complete')

    def _obstacle_near(self) -> bool:
        # Use percentile for side sectors to reduce noisy one-beam spikes.
        return (
            self.sector_min['front'] < self.front_stop_distance
            or self.sector_p['front_left'] < self.side_clearance
            or self.sector_p['front_right'] < self.side_clearance
        )

    def _obstacle_cleared(self) -> bool:
        return (
            self.sector_p['front'] > (self.front_stop_distance + 0.12)
            and self.sector_p['front_left'] > (self.side_clearance + 0.08)
            and self.sector_p['front_right'] > (self.side_clearance + 0.08)
        )

    def _handle_coverage_blocked_skip(self, now: float, target_source: str):
        if target_source != 'coverage':
            self.coverage_blocked_since = None
            return

        if self.sector_min['front'] < self.front_stop_distance:
            if self.coverage_blocked_since is None:
                self.coverage_blocked_since = now
            elif (now - self.coverage_blocked_since) > self.coverage_skip_timeout_sec:
                self.get_logger().warn(
                    f'Waypoint {self.current_waypoint_idx} blocked too long. Skipping.'
                )
                self.current_waypoint_idx += 1
                self.coverage_blocked_since = None
                self.last_coverage_target_odom = None
                self._transition(MotionState.ALIGN_TO_TARGET, 'skip blocked waypoint')
        else:
            self.coverage_blocked_since = None

    def _handle_frontier_blocked_skip(self, now: float, target_source: str) -> bool:
        if target_source != 'frontier':
            self.frontier_blocked_since = None
            return False

        if self.sector_min['front'] < self.front_stop_distance:
            if self.frontier_blocked_since is None:
                self.frontier_blocked_since = now
            elif (now - self.frontier_blocked_since) > self.frontier_blocked_skip_timeout_sec:
                self._blacklist_current_frontier_target('front blocked too long near obstacle')
                self.frontier_blocked_since = None
                self.stop_robot()
                self._transition(MotionState.ALIGN_TO_TARGET, 'skip blocked frontier target')
                return True
        else:
            self.frontier_blocked_since = None

        return False

    def _reset_local_loop_window(self, now: float):
        self.loop_anchor_x = self.pose_x
        self.loop_anchor_y = self.pose_y
        self.loop_anchor_time = now
        self.loop_path_accum = 0.0

    def _local_loop_detected(self, now: float) -> bool:
        if not self.have_pose:
            return False

        elapsed = now - self.loop_anchor_time
        if elapsed < self.local_loop_timeout_sec:
            return False

        displacement = math.hypot(
            self.pose_x - self.loop_anchor_x,
            self.pose_y - self.loop_anchor_y,
        )
        detected = (
            displacement < self.local_loop_radius_m
            and self.loop_path_accum > self.local_loop_min_path_m
        )
        self._reset_local_loop_window(now)
        return detected

    def control_loop(self):
        now = self.now_sec()

        if not self.have_pose:
            self.stop_robot()
            return

        if self.last_scan_time <= 0.0 or (now - self.last_scan_time) > self.scan_timeout_sec:
            self.stop_robot()
            self._transition(MotionState.RECOVER_STUCK, 'scan timeout')
            return

        if self.last_odom_time <= 0.0 or (now - self.last_odom_time) > self.odom_timeout_sec:
            self.stop_robot()
            return

        target, target_source = self._resolve_target()
        if target_source == 'coverage_done':
            self._transition(MotionState.GOAL_REACHED, 'coverage complete')
            self.stop_robot()
            return

        if target_source != self.last_target_source:
            self.get_logger().info(f'Target source -> {target_source}')
            self.last_target_source = target_source

        if self._state_timed_out(now):
            if target_source == 'frontier':
                self._blacklist_current_frontier_target(
                    f'state timeout while in {self.state.value}'
                )
            self._transition(MotionState.RECOVER_STUCK, 'state timeout')

        if self._check_stuck(now) and self.state not in (
            MotionState.BACKUP,
            MotionState.TURN,
            MotionState.RECOVER_STUCK,
        ):
            if target_source == 'frontier':
                self._blacklist_current_frontier_target('stuck while moving to frontier')
            self._transition(MotionState.RECOVER_STUCK, 'stuck detected')

        if self._local_loop_detected(now):
            if target_source == 'frontier':
                self._blacklist_current_frontier_target('local loop detected near same region')
            elif target_source == 'coverage' and self.waypoints:
                self.current_waypoint_idx = min(
                    self.current_waypoint_idx + 3,
                    len(self.waypoints) - 1,
                )
                self.coverage_blocked_since = None
                self.get_logger().warn(
                    'Coverage local loop detected. Fast-forwarding waypoint index.'
                )
            self._transition(MotionState.RECOVER_STUCK, 'local loop escape')
            return

        self._handle_coverage_blocked_skip(now, target_source)
        if self._handle_frontier_blocked_skip(now, target_source):
            return

        front = self.sector_min['front']
        if front < self.emergency_stop_distance and self.state != MotionState.BACKUP:
            self._transition(MotionState.BACKUP, 'emergency stop distance reached')

        if target is None:
            if self.stop_when_coverage_done:
                self._transition(MotionState.GOAL_REACHED, 'no target available')
                self.stop_robot()
                return
            target = self._exploration_target()

        target_distance, target_yaw_error = self._goal_metrics(target)

        obstacle_near = self._obstacle_near()

        if target_distance <= self.goal_tolerance_m:
            if target_source == 'coverage':
                self.current_waypoint_idx += 1
                self.get_logger().info(
                    f'Coverage waypoint {self.current_waypoint_idx - 1} reached.'
                )
            elif target_source == 'frontier':
                self._clear_frontier_target()
            self._transition(MotionState.GOAL_REACHED, 'goal reached')

        if self.state == MotionState.GOAL_REACHED:
            if target_source == 'coverage' and self.stop_when_coverage_done and (
                self.current_waypoint_idx >= len(self.waypoints)
            ):
                self.stop_robot()
                return
            self._transition(MotionState.ALIGN_TO_TARGET, 'continuing mission')

        if self.state == MotionState.BACKUP:
            if now < self.backup_until:
                backup_turn = -0.25 * self.turn_sign * self.max_angular_speed
                self.publish_cmd(-self.backup_speed, backup_turn)
            else:
                self._transition(MotionState.TURN, 'backup complete')
            return

        if self.state == MotionState.TURN:
            elapsed = now - self.state_enter_time
            if (
                front > (self.front_stop_distance + 0.18)
                and elapsed > self.turn_min_duration_sec
            ):
                self._transition(MotionState.ALIGN_TO_TARGET, 'front became clear')
            elif now < self.turn_until:
                self.publish_cmd(0.0, self.turn_sign * self.angular_speed)
            else:
                self._transition(MotionState.ALIGN_TO_TARGET, 'turn timeout reached')
            return

        if self.state == MotionState.RECOVER_STUCK:
            self._run_recover(now)
            return

        if self.state == MotionState.AVOID_OBSTACLE:
            if (now - self.state_enter_time) > 0.35 and self._obstacle_cleared():
                if self.avoid_clear_since is None:
                    self.avoid_clear_since = now
                elif (now - self.avoid_clear_since) > self.avoid_clear_hold_sec:
                    self._transition(MotionState.ALIGN_TO_TARGET, 'obstacle cleared')
            elif now > self.avoid_until:
                self._transition(MotionState.BACKUP, 'avoid timeout')
            else:
                self.avoid_clear_since = None
                turn_sign = self._choose_turn_direction()
                linear = 0.03 if front > (self.front_stop_distance + 0.12) else 0.0
                self.publish_cmd(linear, turn_sign * 0.75 * self.angular_speed)
            return

        if self.state == MotionState.ALIGN_TO_TARGET:
            # While aligning in place, side obstacles are expected in narrow areas.
            # Only trigger avoid if front is truly blocked for forward progress.
            front_blocked_for_align = (
                self.sector_min['front'] < self.align_front_block_distance
            )
            if front_blocked_for_align:
                self._transition(MotionState.AVOID_OBSTACLE, 'front blocked while aligning')
                return

            if abs(target_yaw_error) <= self.align_yaw_tolerance_rad:
                self._transition(MotionState.FORWARD, 'heading aligned')
                return

            align_angular = clamp(
                1.8 * target_yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
            self.publish_cmd(0.0, align_angular)
            return

        if self.state == MotionState.FORWARD:
            if obstacle_near:
                self._transition(MotionState.AVOID_OBSTACLE, 'obstacle ahead in forward')
                return

            if abs(target_yaw_error) > (2.0 * self.align_yaw_tolerance_rad):
                self._transition(MotionState.ALIGN_TO_TARGET, 'large heading error')
                return

            sampled = self._sample_velocity_command(target)
            if sampled is None:
                if self.sector_p['front'] > (self.align_front_block_distance + 0.06):
                    # If sampling fails but front is not critically blocked,
                    # creep forward with a mild turn toward open space.
                    turn_sign = self._choose_turn_direction()
                    self.publish_cmd(0.02, turn_sign * 0.45 * self.angular_speed)
                    return
                self._transition(MotionState.AVOID_OBSTACLE, 'no safe sampled velocity')
                return

            linear, angular = sampled
            if target_distance < self.slowdown_distance_m:
                max_close_speed = clamp(
                    0.25 * target_distance,
                    0.03,
                    self.linear_speed,
                )
                linear = min(linear, max_close_speed)

            self.publish_cmd(linear, angular)
            return

        self.publish_cmd(0.0, 0.0)

    def destroy_node(self):
        try:
            self.stop_robot()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
