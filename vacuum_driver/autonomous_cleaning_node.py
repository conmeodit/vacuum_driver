import heapq
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


class Phase(Enum):
    EXPLORE = 'explore'
    COVER = 'cover'
    DONE = 'done'


class RecoveryStage(Enum):
    NONE = 'none'
    BACKUP = 'backup'
    TURN = 'turn'


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class Target:
    grid: Tuple[int, int]
    world: Tuple[float, float]
    kind: str
    score: float


@dataclass
class ScanSectors:
    front: float = float('inf')
    front_left: float = float('inf')
    front_right: float = float('inf')
    left: float = float('inf')
    right: float = float('inf')
    rear: float = float('inf')


class AutonomousCleaningNode(Node):
    """Frontier exploration followed by visited-cell coverage for a small robot."""

    def __init__(self):
        super().__init__('autonomous_cleaning_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('path_topic', '/autonomy/path')
        self.declare_parameter('marker_topic', '/autonomy/markers')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('replan_period_sec', 1.0)
        self.declare_parameter('tf_lookup_timeout_sec', 0.15)

        self.declare_parameter('robot_radius_m', 0.26)
        self.declare_parameter('obstacle_threshold', 50)
        self.declare_parameter('frontier_min_cluster_size', 8)
        self.declare_parameter('frontier_relaxed_min_cluster_size', 3)
        self.declare_parameter('frontier_min_distance_m', 0.30)
        self.declare_parameter('frontier_gain_weight', 0.035)
        self.declare_parameter('frontier_blacklist_radius_m', 0.45)
        self.declare_parameter('frontier_blacklist_timeout_sec', 35.0)
        self.declare_parameter('map_stable_duration_sec', 8.0)
        self.declare_parameter('exploration_settle_sec', 5.0)
        self.declare_parameter('map_stable_known_delta_cells', 12)
        self.declare_parameter('map_stable_origin_delta_m', 0.06)

        self.declare_parameter('coverage_spacing_m', 0.24)
        self.declare_parameter('coverage_visited_radius_m', 0.24)
        self.declare_parameter('coverage_required_ratio', 0.985)
        self.declare_parameter('coverage_min_target_distance_m', 0.30)
        self.declare_parameter('coverage_gain_weight', 0.020)
        self.declare_parameter('coverage_switch_distance_weight', 0.12)

        self.declare_parameter('goal_tolerance_m', 0.18)
        self.declare_parameter('path_lookahead_m', 0.30)
        self.declare_parameter('waypoint_tolerance_m', 0.10)
        self.declare_parameter('max_linear_speed', 0.12)
        self.declare_parameter('min_linear_speed', 0.035)
        self.declare_parameter('max_angular_speed', 0.80)
        self.declare_parameter('heading_kp', 1.85)

        self.declare_parameter('front_stop_distance_m', 0.34)
        self.declare_parameter('emergency_stop_distance_m', 0.22)
        self.declare_parameter('slowdown_distance_m', 0.75)
        self.declare_parameter('side_clearance_m', 0.23)
        self.declare_parameter('scan_timeout_sec', 1.0)

        self.declare_parameter('stuck_timeout_sec', 5.0)
        self.declare_parameter('stuck_min_progress_m', 0.08)
        self.declare_parameter('target_timeout_min_sec', 16.0)
        self.declare_parameter('target_timeout_speed_factor', 3.2)
        self.declare_parameter('backup_duration_sec', 0.75)
        self.declare_parameter('turn_duration_sec', 1.25)
        self.declare_parameter('backup_speed', 0.055)

        self.map_topic = str(self.get_parameter('map_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.control_rate_hz = max(2.0, float(self.get_parameter('control_rate_hz').value))
        self.replan_period_sec = max(0.25, float(self.get_parameter('replan_period_sec').value))
        self.tf_lookup_timeout_sec = max(
            0.01, float(self.get_parameter('tf_lookup_timeout_sec').value)
        )

        self.robot_radius_m = max(0.05, float(self.get_parameter('robot_radius_m').value))
        self.obstacle_threshold = int(self.get_parameter('obstacle_threshold').value)
        self.frontier_min_cluster_size = max(
            1, int(self.get_parameter('frontier_min_cluster_size').value)
        )
        self.frontier_relaxed_min_cluster_size = max(
            1, int(self.get_parameter('frontier_relaxed_min_cluster_size').value)
        )
        self.frontier_min_distance_m = max(
            0.0, float(self.get_parameter('frontier_min_distance_m').value)
        )
        self.frontier_gain_weight = max(
            0.0, float(self.get_parameter('frontier_gain_weight').value)
        )
        self.frontier_blacklist_radius_m = max(
            0.0, float(self.get_parameter('frontier_blacklist_radius_m').value)
        )
        self.frontier_blacklist_timeout_sec = max(
            1.0, float(self.get_parameter('frontier_blacklist_timeout_sec').value)
        )
        self.map_stable_duration_sec = max(
            1.0, float(self.get_parameter('map_stable_duration_sec').value)
        )
        self.exploration_settle_sec = max(
            1.0, float(self.get_parameter('exploration_settle_sec').value)
        )
        self.map_stable_known_delta_cells = max(
            0, int(self.get_parameter('map_stable_known_delta_cells').value)
        )
        self.map_stable_origin_delta_m = max(
            0.0, float(self.get_parameter('map_stable_origin_delta_m').value)
        )

        self.coverage_spacing_m = max(0.05, float(self.get_parameter('coverage_spacing_m').value))
        self.coverage_visited_radius_m = max(
            0.05, float(self.get_parameter('coverage_visited_radius_m').value)
        )
        self.coverage_required_ratio = clamp(
            float(self.get_parameter('coverage_required_ratio').value), 0.50, 1.0
        )
        self.coverage_min_target_distance_m = max(
            0.0, float(self.get_parameter('coverage_min_target_distance_m').value)
        )
        self.coverage_gain_weight = max(
            0.0, float(self.get_parameter('coverage_gain_weight').value)
        )
        self.coverage_switch_distance_weight = max(
            0.0, float(self.get_parameter('coverage_switch_distance_weight').value)
        )

        self.goal_tolerance_m = max(0.05, float(self.get_parameter('goal_tolerance_m').value))
        self.path_lookahead_m = max(0.05, float(self.get_parameter('path_lookahead_m').value))
        self.waypoint_tolerance_m = max(
            0.03, float(self.get_parameter('waypoint_tolerance_m').value)
        )
        self.max_linear_speed = max(0.02, float(self.get_parameter('max_linear_speed').value))
        self.min_linear_speed = max(0.0, float(self.get_parameter('min_linear_speed').value))
        self.max_angular_speed = max(0.1, float(self.get_parameter('max_angular_speed').value))
        self.heading_kp = max(0.1, float(self.get_parameter('heading_kp').value))

        self.front_stop_distance_m = max(
            0.05, float(self.get_parameter('front_stop_distance_m').value)
        )
        self.emergency_stop_distance_m = max(
            0.05, float(self.get_parameter('emergency_stop_distance_m').value)
        )
        self.slowdown_distance_m = max(
            self.front_stop_distance_m + 0.05,
            float(self.get_parameter('slowdown_distance_m').value),
        )
        self.side_clearance_m = max(0.05, float(self.get_parameter('side_clearance_m').value))
        self.scan_timeout_sec = max(0.1, float(self.get_parameter('scan_timeout_sec').value))

        self.stuck_timeout_sec = max(1.0, float(self.get_parameter('stuck_timeout_sec').value))
        self.stuck_min_progress_m = max(
            0.01, float(self.get_parameter('stuck_min_progress_m').value)
        )
        self.target_timeout_min_sec = max(
            3.0, float(self.get_parameter('target_timeout_min_sec').value)
        )
        self.target_timeout_speed_factor = max(
            1.0, float(self.get_parameter('target_timeout_speed_factor').value)
        )
        self.backup_duration_sec = max(0.1, float(self.get_parameter('backup_duration_sec').value))
        self.turn_duration_sec = max(0.2, float(self.get_parameter('turn_duration_sec').value))
        self.backup_speed = max(0.01, float(self.get_parameter('backup_speed').value))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 20)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.map_msg: Optional[OccupancyGrid] = None
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0
        self.origin_cos = 1.0
        self.origin_sin = 0.0
        self.passable = bytearray()
        self.inflated_obstacles = bytearray()
        self.map_version = 0
        self.known_count = 0
        self.last_known_count = 0
        self.known_stable_since = self.now_sec()
        self.geometry_stable_since = self.now_sec()

        self.phase = Phase.EXPLORE
        self.pose: Optional[Pose2D] = None
        self.last_pose_time = 0.0
        self.sectors = ScanSectors()
        self.last_scan_time = 0.0

        self.target: Optional[Target] = None
        self.path_cells: List[Tuple[int, int]] = []
        self.path_world: List[Tuple[float, float]] = []
        self.path_cursor = 0
        self.planned_map_version = -1
        self.last_replan_time = 0.0
        self.target_started_at = 0.0
        self.target_path_length = 0.0

        self.visited_cells: Set[int] = set()
        self.reachable_cells: Set[int] = set()
        self.last_coverage_ratio = 0.0
        self.last_total_reachable = 0
        self.last_frontier_cells: List[Tuple[int, int]] = []
        self.no_frontier_since: Optional[float] = None
        self.last_coverage_goal: Optional[Tuple[float, float]] = None

        self.blacklisted_targets: Dict[int, float] = {}
        self.recovery_stage = RecoveryStage.NONE
        self.recovery_until = 0.0
        self.recovery_turn_sign = 1.0
        self.recovery_reason = ''

        self.progress_anchor: Optional[Pose2D] = None
        self.progress_anchor_time = self.now_sec()
        self.last_commanded_linear = 0.0
        self.last_marker_time = 0.0
        self.last_status_log_time = 0.0

        self.get_logger().info(
            'Autonomous cleaning ready: '
            'frontier exploration -> visited-cell coverage, '
            f'robot_radius={self.robot_radius_m:.2f}m, '
            f'coverage_radius={self.coverage_visited_radius_m:.2f}m'
        )

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def map_cb(self, msg: OccupancyGrid):
        had_previous_map = self.map_msg is not None and self.width > 0 and self.height > 0
        previous_geometry = self._current_geometry() if had_previous_map else None

        self.map_msg = msg
        self.width = int(msg.info.width)
        self.height = int(msg.info.height)
        self.resolution = float(msg.info.resolution)
        self.origin_x = float(msg.info.origin.position.x)
        self.origin_y = float(msg.info.origin.position.y)
        q = msg.info.origin.orientation
        self.origin_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.origin_cos = math.cos(self.origin_yaw)
        self.origin_sin = math.sin(self.origin_yaw)
        self.map_version += 1

        now = self.now_sec()
        geometry_changed = False
        if previous_geometry is not None:
            geometry_changed = self._geometry_changed(previous_geometry)
            if geometry_changed:
                self.geometry_stable_since = now
        else:
            self.geometry_stable_since = now

        self.known_count = sum(1 for value in msg.data if int(value) >= 0)
        if abs(self.known_count - self.last_known_count) > self.map_stable_known_delta_cells:
            self.known_stable_since = now
            self.last_known_count = self.known_count

        self._build_passability()
        if geometry_changed and previous_geometry is not None:
            self._remap_spatial_state(previous_geometry, now)
        else:
            self._filter_spatial_state()

    def scan_cb(self, msg: LaserScan):
        self.sectors = ScanSectors(
            front=self._sector_min(msg, -18.0, 18.0),
            front_left=self._sector_min(msg, 18.0, 65.0),
            front_right=self._sector_min(msg, -65.0, -18.0),
            left=self._sector_min(msg, 65.0, 120.0),
            right=self._sector_min(msg, -120.0, -65.0),
            rear=min(self._sector_min(msg, 145.0, 180.0), self._sector_min(msg, -180.0, -145.0)),
        )
        self.last_scan_time = self.now_sec()

    def _sector_min(self, msg: LaserScan, start_deg: float, end_deg: float) -> float:
        if msg.angle_increment == 0.0 or len(msg.ranges) == 0:
            return float('inf')

        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)
        angle_min = float(msg.angle_min)
        angle_inc = float(msg.angle_increment)
        i0 = int(math.floor((start_rad - angle_min) / angle_inc))
        i1 = int(math.ceil((end_rad - angle_min) / angle_inc))
        if i0 > i1:
            i0, i1 = i1, i0

        i0 = max(0, i0)
        i1 = min(len(msg.ranges) - 1, i1)
        if i1 < i0:
            return float('inf')

        fallback_max = msg.range_max if math.isfinite(msg.range_max) else 20.0
        values = []
        for index in range(i0, i1 + 1):
            value = float(msg.ranges[index])
            if not math.isfinite(value):
                continue
            if value < msg.range_min or value > fallback_max:
                continue
            values.append(value)

        return min(values) if values else fallback_max

    def _build_passability(self):
        if self.map_msg is None or self.width <= 0 or self.height <= 0:
            return

        total = self.width * self.height
        self.inflated_obstacles = bytearray(total)
        self.passable = bytearray(total)
        inflation_cells = int(math.ceil(self.robot_radius_m / max(self.resolution, 1e-6)))

        obstacle_cells = []
        data = self.map_msg.data
        for index, value in enumerate(data):
            if int(value) >= self.obstacle_threshold:
                obstacle_cells.append(index)

        radius_sq = inflation_cells * inflation_cells
        for index in obstacle_cells:
            cx = index % self.width
            cy = index // self.width
            for dy in range(-inflation_cells, inflation_cells + 1):
                gy = cy + dy
                if gy < 0 or gy >= self.height:
                    continue
                for dx in range(-inflation_cells, inflation_cells + 1):
                    if (dx * dx + dy * dy) > radius_sq:
                        continue
                    gx = cx + dx
                    if gx < 0 or gx >= self.width:
                        continue
                    self.inflated_obstacles[self._index(gx, gy)] = 1

        for index, value in enumerate(data):
            cell_value = int(value)
            if 0 <= cell_value < self.obstacle_threshold and not self.inflated_obstacles[index]:
                self.passable[index] = 1

    def _update_pose(self) -> bool:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
        except TransformException as exc:
            now = self.now_sec()
            if (now - self.last_status_log_time) > 2.0:
                self.get_logger().warn(f'Waiting for TF {self.map_frame}->{self.base_frame}: {exc}')
                self.last_status_log_time = now
            return False

        trans = transform.transform.translation
        rot = transform.transform.rotation
        self.pose = Pose2D(
            float(trans.x),
            float(trans.y),
            yaw_from_quaternion(rot.x, rot.y, rot.z, rot.w),
        )
        self.last_pose_time = self.now_sec()
        return True

    def control_loop(self):
        now = self.now_sec()

        if self.map_msg is None or not self.passable:
            self.stop_robot()
            return
        if not self._update_pose() or self.pose is None:
            self.stop_robot()
            return
        if (now - self.last_scan_time) > self.scan_timeout_sec:
            self.stop_robot()
            return

        self._mark_visited(self.pose)
        self._publish_visualization(now)

        if self.recovery_stage != RecoveryStage.NONE:
            self._run_recovery(now)
            return

        if self.sectors.front < self.emergency_stop_distance_m:
            self._start_recovery('emergency front obstacle')
            self._run_recovery(now)
            return

        self._check_progress(now)
        if self.recovery_stage != RecoveryStage.NONE:
            self._run_recovery(now)
            return

        force_replan = (now - self.last_replan_time) >= self.replan_period_sec
        if force_replan or not self.path_world or self.planned_map_version != self.map_version:
            self._ensure_target_and_path(now, force=force_replan)

        if self.phase == Phase.DONE:
            self.stop_robot()
            return

        if self.target is None or not self.path_world:
            if self.phase == Phase.EXPLORE:
                self.publish_cmd(0.0, 0.35 * self._best_turn_sign())
                return
            self.stop_robot()
            return

        if self._target_timed_out(now):
            self._blacklist_current_target(now, 'target timeout')
            self._clear_navigation()
            self._ensure_target_and_path(now, force=True)
            return

        self._follow_path(now)

    def _ensure_target_and_path(self, now: float, force: bool = False):
        if self.pose is None:
            return

        start = self.world_to_grid(self.pose.x, self.pose.y)
        if start is None or not self._is_passable(start[0], start[1]):
            nearest = self._nearest_passable_to_pose()
            if nearest is None:
                self.stop_robot()
                return
            start = nearest

        if self.target is not None and self._target_reached(self.target):
            if self.target.kind == 'coverage':
                self.last_coverage_goal = self.target.world
            if self.target.kind == 'frontier':
                self._clear_navigation()
            else:
                self._clear_navigation(keep_phase=True)

        if self.target is None:
            self.target = self._select_target(start, now)
            if self.target is not None:
                self.target_started_at = now
                self.progress_anchor = self.pose
                self.progress_anchor_time = now
                self.get_logger().info(
                    f'New {self.target.kind} target: '
                    f'x={self.target.world[0]:.2f}, y={self.target.world[1]:.2f}, '
                    f'phase={self.phase.value}'
                )

        if self.target is None:
            return

        if not force and self.path_world and self.planned_map_version == self.map_version:
            return

        path = self._a_star(start, self.target.grid)
        if not path:
            self._blacklist_current_target(now, 'no path')
            self._clear_navigation()
            return

        self.path_cells = self._smooth_path(path)
        self.path_world = [self.grid_to_world(gx, gy) for gx, gy in self.path_cells]
        self.path_cursor = 0
        self.planned_map_version = self.map_version
        self.last_replan_time = now
        self.target_path_length = self._path_length(self.path_world)
        self._publish_path()

    def _select_target(self, start: Tuple[int, int], now: float) -> Optional[Target]:
        self._expire_blacklist(now)

        if self.phase == Phase.EXPLORE:
            target = self._select_frontier_target(start, now)
            if target is not None:
                self.no_frontier_since = None
                return target

            relaxed_target = self._select_frontier_target(
                start,
                now,
                min_cluster_size=self.frontier_relaxed_min_cluster_size,
            )
            if relaxed_target is not None:
                self.no_frontier_since = None
                return relaxed_target

            if self.no_frontier_since is None:
                self.no_frontier_since = now
                self.get_logger().info('No reachable frontier found; waiting for map to settle.')

            if (
                (now - self.no_frontier_since) >= self.exploration_settle_sec
                and (now - self.known_stable_since) >= self.map_stable_duration_sec
                and (now - self.geometry_stable_since) >= self.map_stable_duration_sec
            ):
                self.phase = Phase.COVER
                self._clear_navigation(keep_phase=True)
                self.last_coverage_goal = None
                self.get_logger().info('Map is stable. Switching to coverage mode.')
                return self._select_coverage_target(start, now)
            return None

        if self.phase == Phase.COVER:
            if (now - self.geometry_stable_since) < self.map_stable_duration_sec:
                target = self._select_frontier_target(
                    start,
                    now,
                    min_cluster_size=self.frontier_relaxed_min_cluster_size,
                )
                if target is not None:
                    self.phase = Phase.EXPLORE
                    self.no_frontier_since = None
                    self.get_logger().info(
                        'Map expanded while covering. Resuming frontier exploration.'
                    )
                    return target
            return self._select_coverage_target(start, now)

        return None

    def _select_frontier_target(
        self,
        start: Tuple[int, int],
        now: float,
        min_cluster_size: Optional[int] = None,
    ) -> Optional[Target]:
        distances = self._distance_field(start)
        if not distances:
            return None

        frontier_mask = self._frontier_mask()
        effective_min_cluster_size = (
            self.frontier_min_cluster_size
            if min_cluster_size is None
            else max(1, min_cluster_size)
        )
        clusters = self._frontier_clusters(frontier_mask, effective_min_cluster_size)
        candidates: List[Target] = []
        self.last_frontier_cells = []

        for cluster in clusters:
            reachable_cells = [cell for cell in cluster if self._index(*cell) in distances]
            if len(reachable_cells) < effective_min_cluster_size:
                continue
            self.last_frontier_cells.extend(reachable_cells[:16])
            best_cell = min(
                reachable_cells,
                key=lambda cell: distances[self._index(*cell)],
            )
            if self._is_blacklisted(best_cell, now):
                continue

            world = self.grid_to_world(best_cell[0], best_cell[1])
            robot_distance = math.hypot(world[0] - self.pose.x, world[1] - self.pose.y)
            if robot_distance < self.frontier_min_distance_m:
                continue

            travel = distances[self._index(*best_cell)] * self.resolution
            gain = math.sqrt(float(len(cluster))) * self.resolution
            score = travel - self.frontier_gain_weight * gain
            candidates.append(Target(best_cell, world, 'frontier', score))

        if not candidates:
            return None

        return min(candidates, key=lambda target: target.score)

    def _frontier_mask(self) -> bytearray:
        total = self.width * self.height
        mask = bytearray(total)
        if self.map_msg is None:
            return mask

        for gy in range(self.height):
            for gx in range(self.width):
                index = self._index(gx, gy)
                if not self.passable[index]:
                    continue
                if self._has_unknown_neighbor(gx, gy):
                    mask[index] = 1
        return mask

    def _frontier_clusters(
        self,
        mask: bytearray,
        min_cluster_size: int,
    ) -> List[List[Tuple[int, int]]]:
        clusters: List[List[Tuple[int, int]]] = []
        visited = bytearray(len(mask))
        for index, is_frontier in enumerate(mask):
            if not is_frontier or visited[index]:
                continue

            gx = index % self.width
            gy = index // self.width
            queue = deque([(gx, gy)])
            visited[index] = 1
            cluster = []

            while queue:
                cx, cy = queue.popleft()
                cluster.append((cx, cy))
                for nx, ny in self._neighbors8(cx, cy):
                    ni = self._index(nx, ny)
                    if mask[ni] and not visited[ni]:
                        visited[ni] = 1
                        queue.append((nx, ny))

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _select_coverage_target(self, start: Tuple[int, int], now: float) -> Optional[Target]:
        distances = self._distance_field(start)
        if not distances:
            return None

        self.reachable_cells = set(distances.keys())
        total_reachable = len(self.reachable_cells)
        if total_reachable == 0:
            return None

        visited_reachable = len(self.visited_cells.intersection(self.reachable_cells))
        self.last_total_reachable = total_reachable
        self.last_coverage_ratio = visited_reachable / float(total_reachable)

        if self.last_coverage_ratio >= self.coverage_required_ratio:
            self.phase = Phase.DONE
            self._clear_navigation(keep_phase=True)
            self.get_logger().info(
                f'Coverage complete: {self.last_coverage_ratio * 100.0:.1f}% reachable cells visited.'
            )
            return None

        stride = max(1, int(round(self.coverage_spacing_m / max(self.resolution, 1e-6))))
        candidates: List[Target] = []
        continuity_anchor = self.last_coverage_goal
        for gy in range(0, self.height, stride):
            for gx in range(0, self.width, stride):
                index = self._index(gx, gy)
                if index not in distances or index in self.visited_cells:
                    continue
                if self._is_blacklisted((gx, gy), now):
                    continue
                world = self.grid_to_world(gx, gy)
                robot_distance = math.hypot(world[0] - self.pose.x, world[1] - self.pose.y)
                if robot_distance < self.coverage_min_target_distance_m:
                    continue
                travel = distances[index] * self.resolution
                gain = self._local_unvisited_gain(gx, gy)
                heading = math.atan2(world[1] - self.pose.y, world[0] - self.pose.x)
                heading_cost = abs(normalize_angle(heading - self.pose.yaw)) * 0.08
                continuity_cost = 0.0
                if continuity_anchor is not None:
                    continuity_cost = (
                        self.coverage_switch_distance_weight
                        * math.hypot(
                            world[0] - continuity_anchor[0],
                            world[1] - continuity_anchor[1],
                        )
                    )
                score = travel + heading_cost + continuity_cost - self.coverage_gain_weight * gain
                candidates.append(Target((gx, gy), world, 'coverage', score))

        if candidates:
            return min(candidates, key=lambda target: target.score)

        # If sparse candidates are exhausted, finish with any remaining reachable cell.
        fallback = []
        for index, travel_cells in distances.items():
            if index in self.visited_cells:
                continue
            gx = index % self.width
            gy = index // self.width
            if self._is_blacklisted((gx, gy), now):
                continue
            world = self.grid_to_world(gx, gy)
            fallback.append(Target((gx, gy), world, 'coverage', travel_cells * self.resolution))

        if fallback:
            return min(fallback, key=lambda target: target.score)

        self.phase = Phase.DONE
        self._clear_navigation(keep_phase=True)
        self.get_logger().info(
            f'Coverage complete: {self.last_coverage_ratio * 100.0:.1f}% reachable cells visited.'
        )
        return None

    def _local_unvisited_gain(self, gx: int, gy: int) -> float:
        radius_cells = max(1, int(math.ceil(self.coverage_visited_radius_m / self.resolution)))
        count = 0
        for dy in range(-radius_cells, radius_cells + 1):
            cy = gy + dy
            if cy < 0 or cy >= self.height:
                continue
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx + dy * dy) > radius_cells * radius_cells:
                    continue
                cx = gx + dx
                if cx < 0 or cx >= self.width:
                    continue
                index = self._index(cx, cy)
                if self.passable[index] and index not in self.visited_cells:
                    count += 1
        return float(count)

    def _distance_field(self, start: Tuple[int, int]) -> Dict[int, int]:
        if not self._is_passable(start[0], start[1]):
            return {}

        start_index = self._index(start[0], start[1])
        distances = {start_index: 0}
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()
            current_distance = distances[self._index(cx, cy)]
            for nx, ny in self._neighbors8(cx, cy):
                ni = self._index(nx, ny)
                if ni in distances or not self.passable[ni]:
                    continue
                if not self._move_allowed(cx, cy, nx, ny):
                    continue
                distances[ni] = current_distance + 1
                queue.append((nx, ny))

        return distances

    def _a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        if not self._is_passable(start[0], start[1]) or not self._is_passable(goal[0], goal[1]):
            return []

        open_heap = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct_path(came_from, current)
            closed.add(current)

            cx, cy = current
            for nx, ny in self._neighbors8(cx, cy):
                if not self._is_passable(nx, ny):
                    continue
                if not self._move_allowed(cx, cy, nx, ny):
                    continue
                neighbor = (nx, ny)
                step_cost = math.sqrt(2.0) if nx != cx and ny != cy else 1.0
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, float('inf')):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + euclidean(neighbor, goal)
                heapq.heappush(open_heap, (priority, neighbor))

        return []

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        anchor_index = 0
        while anchor_index < len(path) - 1:
            next_index = len(path) - 1
            while next_index > anchor_index + 1:
                if self._line_is_passable(path[anchor_index], path[next_index]):
                    break
                next_index -= 1
            smoothed.append(path[next_index])
            anchor_index = next_index
        return smoothed

    def _line_is_passable(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x = x0
        y = y0

        while True:
            if not self._is_passable(x, y):
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _follow_path(self, now: float):
        if self.pose is None or not self.path_world or self.target is None:
            self.stop_robot()
            return

        while self.path_cursor < len(self.path_world) - 1:
            wx, wy = self.path_world[self.path_cursor]
            if math.hypot(wx - self.pose.x, wy - self.pose.y) > self.waypoint_tolerance_m:
                break
            self.path_cursor += 1

        if self._target_reached(self.target):
            self._clear_navigation(keep_phase=True)
            self.stop_robot()
            return

        lookahead = self.path_world[-1]
        for index in range(self.path_cursor, len(self.path_world)):
            wx, wy = self.path_world[index]
            if math.hypot(wx - self.pose.x, wy - self.pose.y) >= self.path_lookahead_m:
                lookahead = (wx, wy)
                break

        dx = lookahead[0] - self.pose.x
        dy = lookahead[1] - self.pose.y
        heading = math.atan2(dy, dx)
        heading_error = normalize_angle(heading - self.pose.yaw)
        distance_to_goal = math.hypot(
            self.target.world[0] - self.pose.x,
            self.target.world[1] - self.pose.y,
        )

        if self.sectors.front < self.front_stop_distance_m and abs(heading_error) < 1.7:
            self._start_recovery('front blocked while following path')
            self._run_recovery(now)
            return

        angular = clamp(
            self.heading_kp * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        if abs(heading_error) > 0.95:
            linear = 0.0
        else:
            heading_scale = clamp(1.0 - abs(heading_error) / 1.05, 0.25, 1.0)
            linear = self.max_linear_speed * heading_scale
            if distance_to_goal < 0.55:
                linear = min(linear, clamp(0.45 * distance_to_goal, self.min_linear_speed, linear))

            if self.sectors.front < self.slowdown_distance_m:
                clearance_scale = clamp(
                    (self.sectors.front - self.front_stop_distance_m)
                    / max(self.slowdown_distance_m - self.front_stop_distance_m, 1e-6),
                    0.15,
                    1.0,
                )
                linear *= clearance_scale

            linear = clamp(linear, self.min_linear_speed, self.max_linear_speed)

        angular += self._side_clearance_correction()
        angular = clamp(angular, -self.max_angular_speed, self.max_angular_speed)
        self.publish_cmd(linear, angular)

    def _side_clearance_correction(self) -> float:
        correction = 0.0
        if self.sectors.left < self.side_clearance_m:
            correction -= 1.2 * (self.side_clearance_m - self.sectors.left)
        if self.sectors.right < self.side_clearance_m:
            correction += 1.2 * (self.side_clearance_m - self.sectors.right)
        return correction

    def _check_progress(self, now: float):
        if self.pose is None or self.target is None:
            self.progress_anchor = self.pose
            self.progress_anchor_time = now
            return

        if self.progress_anchor is None:
            self.progress_anchor = self.pose
            self.progress_anchor_time = now
            return

        moved = math.hypot(
            self.pose.x - self.progress_anchor.x,
            self.pose.y - self.progress_anchor.y,
        )
        if moved >= self.stuck_min_progress_m:
            self.progress_anchor = self.pose
            self.progress_anchor_time = now
            return

        if self.last_commanded_linear > 0.02 and (now - self.progress_anchor_time) > self.stuck_timeout_sec:
            self._start_recovery('stuck: commanded forward but pose barely changed')

    def _target_timed_out(self, now: float) -> bool:
        if self.target is None:
            return False
        expected = self.target_timeout_speed_factor * (
            self.target_path_length / max(self.max_linear_speed, 1e-6)
        )
        timeout = max(self.target_timeout_min_sec, expected)
        return (now - self.target_started_at) > timeout

    def _start_recovery(self, reason: str):
        if self.recovery_stage != RecoveryStage.NONE:
            return

        now = self.now_sec()
        self.recovery_reason = reason
        self.recovery_turn_sign = self._best_turn_sign()

        if self.sectors.rear > self.front_stop_distance_m:
            self.recovery_stage = RecoveryStage.BACKUP
            self.recovery_until = now + self.backup_duration_sec
        else:
            self.recovery_stage = RecoveryStage.TURN
            self.recovery_until = now + self.turn_duration_sec

        self.get_logger().warn(f'Recovery started: {reason}')

    def _run_recovery(self, now: float):
        if self.recovery_stage == RecoveryStage.BACKUP:
            if now < self.recovery_until and self.sectors.rear > self.emergency_stop_distance_m:
                self.publish_cmd(-self.backup_speed, 0.30 * self.recovery_turn_sign)
                return
            self.recovery_stage = RecoveryStage.TURN
            self.recovery_until = now + self.turn_duration_sec

        if self.recovery_stage == RecoveryStage.TURN:
            if now < self.recovery_until:
                self.publish_cmd(0.0, self.recovery_turn_sign * 0.75 * self.max_angular_speed)
                return

        self.stop_robot()
        self._blacklist_current_target(now, self.recovery_reason)
        self._clear_navigation(keep_phase=True)
        self.recovery_stage = RecoveryStage.NONE
        self.recovery_reason = ''
        self.last_replan_time = 0.0

    def _best_turn_sign(self) -> float:
        left_score = min(self.sectors.left, self.sectors.front_left)
        right_score = min(self.sectors.right, self.sectors.front_right)
        return 1.0 if left_score >= right_score else -1.0

    def _target_reached(self, target: Target) -> bool:
        if self.pose is None:
            return False
        return (
            math.hypot(target.world[0] - self.pose.x, target.world[1] - self.pose.y)
            <= self.goal_tolerance_m
        )

    def _blacklist_current_target(self, now: float, reason: str):
        if self.target is None:
            return
        index = self._index(self.target.grid[0], self.target.grid[1])
        self.blacklisted_targets[index] = now + self.frontier_blacklist_timeout_sec
        self.get_logger().warn(
            f'Blacklisting {self.target.kind} target for {self.frontier_blacklist_timeout_sec:.0f}s: '
            f'{reason}'
        )

    def _is_blacklisted(self, cell: Tuple[int, int], now: float) -> bool:
        radius_cells = int(math.ceil(self.frontier_blacklist_radius_m / max(self.resolution, 1e-6)))
        for index, expires_at in self.blacklisted_targets.items():
            if expires_at <= now:
                continue
            bx = index % self.width
            by = index // self.width
            if math.hypot(float(cell[0] - bx), float(cell[1] - by)) <= radius_cells:
                return True
        return False

    def _expire_blacklist(self, now: float):
        expired = [index for index, expires_at in self.blacklisted_targets.items() if expires_at <= now]
        for index in expired:
            del self.blacklisted_targets[index]

    def _clear_navigation(self, keep_phase: bool = False):
        self.target = None
        self.path_cells = []
        self.path_world = []
        self.path_cursor = 0
        self.planned_map_version = -1
        self.target_started_at = 0.0
        self.target_path_length = 0.0
        if not keep_phase:
            self.last_frontier_cells = []

    def _mark_visited(self, pose: Pose2D):
        grid = self.world_to_grid(pose.x, pose.y)
        if grid is None:
            return
        radius_cells = max(1, int(math.ceil(self.coverage_visited_radius_m / self.resolution)))
        gx, gy = grid
        for dy in range(-radius_cells, radius_cells + 1):
            cy = gy + dy
            if cy < 0 or cy >= self.height:
                continue
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx + dy * dy) > radius_cells * radius_cells:
                    continue
                cx = gx + dx
                if cx < 0 or cx >= self.width:
                    continue
                index = self._index(cx, cy)
                if self.passable[index]:
                    self.visited_cells.add(index)

    def publish_cmd(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = float(clamp(linear, -self.max_linear_speed, self.max_linear_speed))
        msg.angular.z = float(clamp(angular, -self.max_angular_speed, self.max_angular_speed))
        self.cmd_pub.publish(msg)
        self.last_commanded_linear = msg.linear.x

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def _publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for wx, wy in self.path_world:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.02
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def _publish_visualization(self, now: float):
        if self.pose is None or (now - self.last_marker_time) < 0.4:
            return
        self.last_marker_time = now

        marker_array = MarkerArray()
        marker_array.markers.extend(self._robot_model_markers())
        marker_array.markers.append(self._target_marker())
        marker_array.markers.append(self._visited_marker())
        marker_array.markers.append(self._frontier_marker())
        marker_array.markers.append(self._status_marker())
        self.marker_pub.publish(marker_array)

    def _base_marker(self, marker_id: int, marker_type: int, ns: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = 1
        return marker

    def _robot_model_markers(self) -> List[Marker]:
        if self.pose is None:
            return []

        markers = []
        qx, qy, qz, qw = quaternion_from_yaw(self.pose.yaw)
        body = self._base_marker(1, Marker.CUBE, 'robot_model')
        body.pose.position.x = self.pose.x
        body.pose.position.y = self.pose.y
        body.pose.position.z = 0.03
        body.pose.orientation.x = qx
        body.pose.orientation.y = qy
        body.pose.orientation.z = qz
        body.pose.orientation.w = qw
        body.scale.x = 0.30
        body.scale.y = 0.40
        body.scale.z = 0.06
        self._set_color(body, 1.0, 0.55, 0.18, 0.95)
        markers.append(body)

        wheel_offsets = [
            (0.10, 0.21),
            (0.10, -0.21),
            (-0.10, 0.21),
            (-0.10, -0.21),
        ]
        for idx, (ox, oy) in enumerate(wheel_offsets, start=2):
            wx = self.pose.x + ox * math.cos(self.pose.yaw) - oy * math.sin(self.pose.yaw)
            wy = self.pose.y + ox * math.sin(self.pose.yaw) + oy * math.cos(self.pose.yaw)
            wheel = self._base_marker(idx, Marker.CUBE, 'robot_model')
            wheel.pose.position.x = wx
            wheel.pose.position.y = wy
            wheel.pose.position.z = 0.045
            wheel.pose.orientation.x = qx
            wheel.pose.orientation.y = qy
            wheel.pose.orientation.z = qz
            wheel.pose.orientation.w = qw
            wheel.scale.x = 0.09
            wheel.scale.y = 0.035
            wheel.scale.z = 0.09
            self._set_color(wheel, 0.03, 0.03, 0.03, 1.0)
            markers.append(wheel)

        lidar_x = self.pose.x - 0.10 * math.cos(self.pose.yaw)
        lidar_y = self.pose.y - 0.10 * math.sin(self.pose.yaw)
        lidar = self._base_marker(6, Marker.CYLINDER, 'robot_model')
        lidar.pose.position.x = lidar_x
        lidar.pose.position.y = lidar_y
        lidar.pose.position.z = 0.105
        lidar.scale.x = 0.12
        lidar.scale.y = 0.12
        lidar.scale.z = 0.05
        self._set_color(lidar, 0.05, 0.16, 0.24, 1.0)
        markers.append(lidar)

        footprint = self._base_marker(7, Marker.LINE_STRIP, 'robot_model')
        footprint.scale.x = 0.025
        self._set_color(footprint, 0.0, 0.9, 0.8, 0.95)
        half_x = 0.5 * 0.30 + 0.03
        half_y = 0.5 * 0.40 + 0.03
        corners = [
            (half_x, half_y),
            (half_x, -half_y),
            (-half_x, -half_y),
            (-half_x, half_y),
            (half_x, half_y),
        ]
        for ox, oy in corners:
            point = Point()
            point.x = self.pose.x + ox * math.cos(self.pose.yaw) - oy * math.sin(self.pose.yaw)
            point.y = self.pose.y + ox * math.sin(self.pose.yaw) + oy * math.cos(self.pose.yaw)
            point.z = 0.025
            footprint.points.append(point)
        markers.append(footprint)
        return markers

    def _target_marker(self) -> Marker:
        marker = self._base_marker(20, Marker.SPHERE, 'autonomy_target')
        if self.target is None:
            marker.action = Marker.DELETE
            return marker
        marker.pose.position.x = self.target.world[0]
        marker.pose.position.y = self.target.world[1]
        marker.pose.position.z = 0.12
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        if self.target.kind == 'frontier':
            self._set_color(marker, 0.05, 0.65, 1.0, 0.95)
        else:
            self._set_color(marker, 0.10, 0.90, 0.30, 0.95)
        return marker

    def _visited_marker(self) -> Marker:
        marker = self._base_marker(30, Marker.CUBE_LIST, 'visited_cells')
        marker.scale.x = max(self.resolution, 0.03)
        marker.scale.y = max(self.resolution, 0.03)
        marker.scale.z = 0.015
        self._set_color(marker, 0.10, 0.85, 0.35, 0.28)
        if not self.visited_cells:
            return marker
        stride = max(1, len(self.visited_cells) // 2500)
        for count, index in enumerate(sorted(self.visited_cells)):
            if count % stride != 0:
                continue
            gx = index % self.width
            gy = index // self.width
            wx, wy = self.grid_to_world(gx, gy)
            point = Point()
            point.x = wx
            point.y = wy
            point.z = 0.012
            marker.points.append(point)
        return marker

    def _frontier_marker(self) -> Marker:
        marker = self._base_marker(40, Marker.CUBE_LIST, 'frontier_cells')
        marker.scale.x = max(self.resolution, 0.04)
        marker.scale.y = max(self.resolution, 0.04)
        marker.scale.z = 0.03
        self._set_color(marker, 0.0, 0.55, 1.0, 0.45)
        for gx, gy in self.last_frontier_cells[:800]:
            wx, wy = self.grid_to_world(gx, gy)
            point = Point()
            point.x = wx
            point.y = wy
            point.z = 0.035
            marker.points.append(point)
        return marker

    def _status_marker(self) -> Marker:
        marker = self._base_marker(50, Marker.TEXT_VIEW_FACING, 'autonomy_status')
        if self.pose is None:
            marker.action = Marker.DELETE
            return marker
        marker.pose.position.x = self.pose.x
        marker.pose.position.y = self.pose.y
        marker.pose.position.z = 0.42
        marker.scale.z = 0.16
        coverage = self.last_coverage_ratio * 100.0
        marker.text = f'{self.phase.value.upper()}  {coverage:.1f}%'
        self._set_color(marker, 1.0, 1.0, 1.0, 0.95)
        return marker

    def _set_color(self, marker: Marker, red: float, green: float, blue: float, alpha: float):
        marker.color.r = float(red)
        marker.color.g = float(green)
        marker.color.b = float(blue)
        marker.color.a = float(alpha)

    def _path_length(self, path: Sequence[Tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for index in range(1, len(path)):
            total += math.hypot(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
        return total

    def _current_geometry(self) -> Dict[str, float]:
        return {
            'width': float(self.width),
            'height': float(self.height),
            'resolution': float(self.resolution),
            'origin_x': float(self.origin_x),
            'origin_y': float(self.origin_y),
            'origin_yaw': float(self.origin_yaw),
            'origin_cos': float(self.origin_cos),
            'origin_sin': float(self.origin_sin),
        }

    def _geometry_changed(self, previous: Dict[str, float]) -> bool:
        if int(previous['width']) != self.width or int(previous['height']) != self.height:
            return True
        if abs(previous['resolution'] - self.resolution) > 1e-9:
            return True
        if abs(previous['origin_x'] - self.origin_x) > self.map_stable_origin_delta_m:
            return True
        if abs(previous['origin_y'] - self.origin_y) > self.map_stable_origin_delta_m:
            return True
        yaw_delta = abs(normalize_angle(previous['origin_yaw'] - self.origin_yaw))
        return yaw_delta > math.radians(1.0)

    def _grid_to_world_in_geometry(
        self,
        gx: int,
        gy: int,
        geometry: Dict[str, float],
    ) -> Tuple[float, float]:
        local_x = (float(gx) + 0.5) * geometry['resolution']
        local_y = (float(gy) + 0.5) * geometry['resolution']
        world_x = (
            geometry['origin_x']
            + geometry['origin_cos'] * local_x
            - geometry['origin_sin'] * local_y
        )
        world_y = (
            geometry['origin_y']
            + geometry['origin_sin'] * local_x
            + geometry['origin_cos'] * local_y
        )
        return world_x, world_y

    def _remap_spatial_state(self, previous_geometry: Dict[str, float], now: float):
        old_width = int(previous_geometry['width'])
        old_height = int(previous_geometry['height'])
        old_total = old_width * old_height

        remapped_visited: Set[int] = set()
        for index in self.visited_cells:
            if index < 0 or index >= old_total:
                continue
            gx = index % old_width
            gy = index // old_width
            world_x, world_y = self._grid_to_world_in_geometry(gx, gy, previous_geometry)
            mapped = self.world_to_grid(world_x, world_y)
            if mapped is None:
                continue
            mapped_index = self._index(mapped[0], mapped[1])
            if self.passable[mapped_index]:
                remapped_visited.add(mapped_index)
        self.visited_cells = remapped_visited

        remapped_blacklist: Dict[int, float] = {}
        for index, expires_at in self.blacklisted_targets.items():
            if expires_at <= now:
                continue
            if index < 0 or index >= old_total:
                continue
            gx = index % old_width
            gy = index // old_width
            world_x, world_y = self._grid_to_world_in_geometry(gx, gy, previous_geometry)
            mapped = self.world_to_grid(world_x, world_y)
            if mapped is None:
                continue
            mapped_index = self._index(mapped[0], mapped[1])
            remapped_blacklist[mapped_index] = max(
                remapped_blacklist.get(mapped_index, 0.0),
                expires_at,
            )
        self.blacklisted_targets = remapped_blacklist

        if self.target is not None:
            mapped = self.world_to_grid(self.target.world[0], self.target.world[1])
            if mapped is None or not self._is_passable(mapped[0], mapped[1]):
                self.target = None
            else:
                self.target.grid = mapped

        self.path_cells = []
        self.path_world = []
        self.path_cursor = 0
        self.planned_map_version = -1
        self.last_replan_time = 0.0
        self.last_frontier_cells = []

    def _filter_spatial_state(self):
        total = self.width * self.height
        filtered_visited = set()
        for index in self.visited_cells:
            if 0 <= index < total and self.passable[index]:
                filtered_visited.add(index)
        self.visited_cells = filtered_visited

        filtered_blacklist = {}
        for index, expires_at in self.blacklisted_targets.items():
            if 0 <= index < total:
                filtered_blacklist[index] = expires_at
        self.blacklisted_targets = filtered_blacklist

    def _nearest_passable_to_pose(self) -> Optional[Tuple[int, int]]:
        if self.pose is None:
            return None
        start = self.world_to_grid(self.pose.x, self.pose.y)
        if start is None:
            return None
        sx, sy = start
        max_radius = max(self.width, self.height)
        for radius in range(1, max_radius):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    gx = sx + dx
                    gy = sy + dy
                    if self._is_passable(gx, gy):
                        return gx, gy
        return None

    def _has_unknown_neighbor(self, gx: int, gy: int) -> bool:
        if self.map_msg is None:
            return False
        for nx, ny in self._neighbors8(gx, gy):
            if int(self.map_msg.data[self._index(nx, ny)]) < 0:
                return True
        return False

    def _neighbors8(self, gx: int, gy: int):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    yield nx, ny

    def _move_allowed(self, gx: int, gy: int, nx: int, ny: int) -> bool:
        if not self._is_passable(nx, ny):
            return False
        if gx != nx and gy != ny:
            return self._is_passable(nx, gy) and self._is_passable(gx, ny)
        return True

    def _is_passable(self, gx: int, gy: int) -> bool:
        if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
            return False
        return bool(self.passable[self._index(gx, gy)])

    def _index(self, gx: int, gy: int) -> int:
        return gy * self.width + gx

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.width <= 0 or self.height <= 0:
            return None
        dx = x - self.origin_x
        dy = y - self.origin_y
        local_x = self.origin_cos * dx + self.origin_sin * dy
        local_y = -self.origin_sin * dx + self.origin_cos * dy
        gx = int(math.floor(local_x / self.resolution))
        gy = int(math.floor(local_y / self.resolution))
        if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        local_x = (float(gx) + 0.5) * self.resolution
        local_y = (float(gy) + 0.5) * self.resolution
        world_x = self.origin_x + self.origin_cos * local_x - self.origin_sin * local_y
        world_y = self.origin_y + self.origin_sin * local_x + self.origin_cos * local_y
        return world_x, world_y

    def destroy_node(self):
        try:
            self.stop_robot()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousCleaningNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
