import math
from collections import deque
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient

    NAV2_AVAILABLE = True
except Exception:
    NavigateToPose = None
    ActionClient = None
    NAV2_AVAILABLE = False


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class CoveragePlannerNode(Node):
    def __init__(self):
        super().__init__('coverage_planner_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('waypoint_topic', '/coverage_waypoints')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('grid_stride_cells', 4)
        self.declare_parameter('obstacle_threshold', 50)
        self.declare_parameter('unknown_is_obstacle', True)
        self.declare_parameter('obstacle_inflation_cells', 2)
        self.declare_parameter('visited_radius_m', 0.18)
        self.declare_parameter('path_sampling_stride_cells', 3)
        self.declare_parameter('max_waypoints_per_plan', 80)

        self.declare_parameter('auto_plan_on_first_map', True)
        self.declare_parameter('auto_replan', True)
        self.declare_parameter('replan_period_sec', 2.0)
        self.declare_parameter('replan_min_robot_shift_m', 0.20)
        self.declare_parameter('auto_start_execution', True)
        self.declare_parameter('use_nav2', True)

        self.declare_parameter('goal_tolerance_m', 0.20)
        self.declare_parameter('linear_speed', 0.08)
        self.declare_parameter('angular_speed', 0.50)
        self.declare_parameter('yaw_align_threshold_rad', 0.35)

        self.declare_parameter('front_stop_distance', 0.30)
        self.declare_parameter('blocked_skip_timeout_sec', 2.5)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('tf_lookup_timeout_sec', 0.2)

        self.map_topic = str(self.get_parameter('map_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.waypoint_topic = str(self.get_parameter('waypoint_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.grid_stride_cells = max(1, int(self.get_parameter('grid_stride_cells').value))
        self.obstacle_threshold = int(self.get_parameter('obstacle_threshold').value)
        self.unknown_is_obstacle = bool(self.get_parameter('unknown_is_obstacle').value)
        self.obstacle_inflation_cells = max(
            0, int(self.get_parameter('obstacle_inflation_cells').value)
        )
        self.visited_radius_m = max(
            0.05, float(self.get_parameter('visited_radius_m').value)
        )
        self.path_sampling_stride_cells = max(
            1, int(self.get_parameter('path_sampling_stride_cells').value)
        )
        self.max_waypoints_per_plan = max(
            1, int(self.get_parameter('max_waypoints_per_plan').value)
        )

        self.auto_plan_on_first_map = bool(
            self.get_parameter('auto_plan_on_first_map').value
        )
        self.auto_replan = bool(self.get_parameter('auto_replan').value)
        self.replan_period_sec = max(
            0.5, float(self.get_parameter('replan_period_sec').value)
        )
        self.replan_min_robot_shift_m = max(
            0.05, float(self.get_parameter('replan_min_robot_shift_m').value)
        )
        self.auto_start_execution = bool(
            self.get_parameter('auto_start_execution').value
        )
        self.use_nav2 = bool(self.get_parameter('use_nav2').value)

        self.goal_tolerance_m = max(0.05, float(self.get_parameter('goal_tolerance_m').value))
        self.linear_speed = max(0.02, float(self.get_parameter('linear_speed').value))
        self.angular_speed = max(0.10, float(self.get_parameter('angular_speed').value))
        self.yaw_align_threshold_rad = max(
            0.05, float(self.get_parameter('yaw_align_threshold_rad').value)
        )

        self.front_stop_distance = max(
            0.10, float(self.get_parameter('front_stop_distance').value)
        )
        self.blocked_skip_timeout_sec = max(
            0.5, float(self.get_parameter('blocked_skip_timeout_sec').value)
        )
        self.control_rate_hz = max(2.0, float(self.get_parameter('control_rate_hz').value))
        self.tf_lookup_timeout_sec = max(
            0.01, float(self.get_parameter('tf_lookup_timeout_sec').value)
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 20)

        self.waypoint_pub = self.create_publisher(PoseArray, self.waypoint_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.map_msg = None
        self.plan_ready = False
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.exec_mode = 'idle'
        self.visited_cells = set()
        self.last_plan_time = 0.0
        self.last_plan_pose = None
        self.last_remaining_cell_count = -1

        self.have_odom = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        self.front_distance = float('inf')
        self.blocked_since = None

        self.nav2_client = None
        if self.use_nav2:
            if NAV2_AVAILABLE:
                self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
            else:
                self.use_nav2 = False
                self.get_logger().warn(
                    'nav2_msgs is not available. Falling back to cmd_vel coverage execution.'
                )

        self.get_logger().info(
            'Coverage planner started: '
            f'map={self.map_topic}, waypoints={self.waypoint_topic}, '
            f'mode={"nav2" if self.use_nav2 else "cmd_vel"}'
        )

    def now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def odom_cb(self, msg):
        self.have_odom = True
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.odom_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def scan_cb(self, msg):
        self.front_distance = self._sector_min(msg, -18.0, 18.0)

    def map_cb(self, msg):
        self.map_msg = msg
        self._maybe_replan(force=not self.plan_ready)

    def _sector_min(self, msg, start_deg, end_deg):
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
        for i in range(i0, i1 + 1):
            value = float(msg.ranges[i])
            if not math.isfinite(value):
                continue
            if value < msg.range_min:
                continue
            if value > fallback_max:
                continue
            values.append(value)

        if not values:
            return fallback_max
        return min(values)

    def _inflate_obstacles(self, occupied_cells, width, height):
        if self.obstacle_inflation_cells <= 0:
            return set(occupied_cells)

        inflated = set(occupied_cells)
        radius = self.obstacle_inflation_cells
        radius_sq = radius * radius

        for ox, oy in occupied_cells:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if (dx * dx + dy * dy) > radius_sq:
                        continue
                    x = ox + dx
                    y = oy + dy
                    if x < 0 or y < 0 or x >= width or y >= height:
                        continue
                    inflated.add((x, y))
        return inflated

    def _world_to_grid(self, x: float, y: float):
        if self.map_msg is None:
            return None

        info = self.map_msg.info
        resolution = float(info.resolution)
        if resolution <= 0.0:
            return None

        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        gx = int(math.floor((x - origin_x) / resolution))
        gy = int(math.floor((y - origin_y) / resolution))
        if gx < 0 or gy < 0 or gx >= int(info.width) or gy >= int(info.height):
            return None
        return gx, gy

    def _grid_to_world(self, x: int, y: int):
        info = self.map_msg.info
        resolution = float(info.resolution)
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        return (
            origin_x + (x + 0.5) * resolution,
            origin_y + (y + 0.5) * resolution,
        )

    def _robot_cell_in_map(self):
        robot_pose = self._robot_pose_in_map()
        if robot_pose is None:
            return None
        return self._world_to_grid(robot_pose[0], robot_pose[1])

    def _mark_robot_visited(self):
        if self.map_msg is None:
            return

        robot_cell = self._robot_cell_in_map()
        if robot_cell is None:
            return

        info = self.map_msg.info
        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)
        data = self.map_msg.data

        radius_cells = max(1, int(math.ceil(self.visited_radius_m / max(resolution, 1e-6))))
        radius_sq = radius_cells * radius_cells
        rx, ry = robot_cell

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx + dy * dy) > radius_sq:
                    continue
                gx = rx + dx
                gy = ry + dy
                if gx < 0 or gy < 0 or gx >= width or gy >= height:
                    continue
                idx = gy * width + gx
                if int(data[idx]) == 0:
                    self.visited_cells.add((gx, gy))

    def _nearest_traversable_cell(self, start_cell, traversable_cells, width, height):
        if start_cell in traversable_cells:
            return start_cell

        sx, sy = start_cell
        max_radius = max(width, height)
        for radius in range(1, max_radius):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    gx = sx + dx
                    gy = sy + dy
                    if gx < 0 or gy < 0 or gx >= width or gy >= height:
                        continue
                    if (gx, gy) in traversable_cells:
                        return gx, gy
        return None

    def _reconstruct_path(self, parents, goal_cell):
        path = []
        current = goal_cell
        while current is not None:
            path.append(current)
            current = parents.get(current)
        path.reverse()
        return path

    def _sample_path_cells(self, path_cells):
        if not path_cells:
            return []

        stride = self.path_sampling_stride_cells
        sampled = [path_cells[0]]
        for idx in range(stride, len(path_cells), stride):
            sampled.append(path_cells[idx])
        if sampled[-1] != path_cells[-1]:
            sampled.append(path_cells[-1])
        return sampled[: self.max_waypoints_per_plan]

    def _maybe_replan(self, force: bool = False):
        if self.map_msg is None or not self.auto_plan_on_first_map:
            return

        self._mark_robot_visited()

        now = self.now_sec()
        if not force:
            if not self.auto_replan:
                return
            if (now - self.last_plan_time) < self.replan_period_sec:
                return
            robot_pose = self._robot_pose_in_map()
            if (
                self.plan_ready
                and self.last_plan_pose is not None
                and robot_pose is not None
                and math.hypot(
                    robot_pose[0] - self.last_plan_pose[0],
                    robot_pose[1] - self.last_plan_pose[1],
                )
                < self.replan_min_robot_shift_m
            ):
                return

        count = self.generate_coverage_waypoints()
        if count > 0 and self.auto_start_execution and self.exec_mode == 'idle':
            self.start_execution()

    def generate_coverage_waypoints(self):
        if self.map_msg is None:
            self.get_logger().warn('Cannot plan coverage: /map is not available yet.')
            return 0

        self._mark_robot_visited()

        info = self.map_msg.info
        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        data = self.map_msg.data

        occupied_cells = set()
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                value = int(data[idx])
                if value < 0:
                    if self.unknown_is_obstacle:
                        occupied_cells.add((x, y))
                    continue
                if value >= self.obstacle_threshold:
                    occupied_cells.add((x, y))

        inflated_occupied = self._inflate_obstacles(occupied_cells, width, height)
        traversable_cells = set()
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if int(data[idx]) != 0:
                    continue
                if (x, y) in inflated_occupied:
                    continue
                traversable_cells.add((x, y))

        robot_cell = self._robot_cell_in_map()
        if robot_cell is None and traversable_cells:
            robot_cell = min(
                traversable_cells,
                key=lambda cell: ((cell[0] - (width * 0.5)) ** 2) + ((cell[1] - (height * 0.5)) ** 2),
            )

        if robot_cell is None:
            self.get_logger().warn('Coverage planning cannot localize robot in map yet.')
            self.plan_ready = False
            self.waypoints = []
            self.current_waypoint_idx = 0
            self.publish_waypoint_markers()
            return 0

        robot_cell = self._nearest_traversable_cell(
            robot_cell, traversable_cells, width, height
        )
        if robot_cell is None:
            self.get_logger().warn('Coverage planning found no traversable free cell near robot.')
            self.plan_ready = False
            self.waypoints = []
            self.current_waypoint_idx = 0
            self.publish_waypoint_markers()
            return 0

        candidate_stride = max(1, self.grid_stride_cells)
        queue = deque([robot_cell])
        parents = {robot_cell: None}
        distances = {robot_cell: 0}
        farthest_goal = None
        farthest_distance = -1
        reachable_unvisited = 0
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        while queue:
            cx, cy = queue.popleft()
            dist = distances[(cx, cy)]

            if (cx, cy) not in self.visited_cells and (dist % candidate_stride) == 0:
                farthest_goal = (cx, cy)
                farthest_distance = dist
                reachable_unvisited += 1

            for dx, dy in neighbors:
                nx = cx + dx
                ny = cy + dy
                neighbor = (nx, ny)
                if neighbor in parents:
                    continue
                if neighbor not in traversable_cells:
                    continue
                parents[neighbor] = (cx, cy)
                distances[neighbor] = dist + 1
                queue.append(neighbor)

        if farthest_goal is None:
            self.plan_ready = False
            self.waypoints = []
            self.current_waypoint_idx = 0
            self.blocked_since = None
            self.publish_waypoint_markers()
            remaining = 0
            if self.last_remaining_cell_count != remaining:
                self.get_logger().info('Coverage planner: known free space is already covered.')
                self.last_remaining_cell_count = remaining
            self.last_plan_time = self.now_sec()
            robot_pose = self._robot_pose_in_map()
            self.last_plan_pose = robot_pose[:2] if robot_pose is not None else None
            return 0

        path_cells = self._reconstruct_path(parents, farthest_goal)
        sampled_cells = self._sample_path_cells(path_cells)

        waypoints = []
        for i, (cx, cy) in enumerate(sampled_cells):
            wx = origin_x + (cx + 0.5) * resolution
            wy = origin_y + (cy + 0.5) * resolution

            if i + 1 < len(sampled_cells):
                nx, ny = sampled_cells[i + 1]
                nx, ny = self._grid_to_world(nx, ny)
                yaw = math.atan2(ny - wy, nx - wx)
            elif i > 0:
                px, py = sampled_cells[i - 1]
                px, py = self._grid_to_world(px, py)
                yaw = math.atan2(wy - py, wx - px)
            else:
                yaw = 0.0

            qx, qy, qz, qw = quaternion_from_yaw(yaw)

            pose = PoseStamped()
            pose.header.frame_id = self.map_frame
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            waypoints.append(pose)

        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.plan_ready = True
        self.exec_mode = 'idle'
        self.blocked_since = None
        self.last_plan_time = self.now_sec()
        robot_pose = self._robot_pose_in_map()
        self.last_plan_pose = robot_pose[:2] if robot_pose is not None else None
        self.last_remaining_cell_count = reachable_unvisited

        self.publish_waypoint_markers()
        self.get_logger().info(
            'Coverage plan ready: '
            f'path_waypoints={len(self.waypoints)}, '
            f'path_cells={len(path_cells)}, '
            f'remaining_samples={reachable_unvisited}, '
            f'farthest_distance={farthest_distance * resolution:.2f}m'
        )
        return len(self.waypoints)

    def publish_waypoint_markers(self):
        marker = PoseArray()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.poses = [pose.pose for pose in self.waypoints]
        self.waypoint_pub.publish(marker)

    def start_execution(self):
        if not self.plan_ready or not self.waypoints:
            self.get_logger().warn('Cannot start coverage: no plan available.')
            return

        if self.use_nav2 and self.nav2_client is not None:
            if self.nav2_client.wait_for_server(timeout_sec=2.0):
                self.exec_mode = 'nav2'
                self.get_logger().info('Coverage execution mode: Nav2 NavigateToPose.')
                self._send_next_nav2_goal()
                return
            self.get_logger().warn('Nav2 server not available. Switching to cmd_vel mode.')

        self.exec_mode = 'cmd_vel'
        self.get_logger().info('Coverage execution mode: cmd_vel fallback controller.')

    def stop_execution(self):
        self.exec_mode = 'done'
        self._publish_cmd(0.0, 0.0)
        self.get_logger().info('Coverage execution complete. Robot stopped.')

    def _send_next_nav2_goal(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            self.stop_execution()
            return

        goal_pose = self.waypoints[self.current_waypoint_idx]
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        future = self.nav2_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_nav2_goal_response)

    def _on_nav2_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Nav2 goal request failed: {exc}')
            self.current_waypoint_idx += 1
            self._send_next_nav2_goal()
            return

        if not goal_handle.accepted:
            self.get_logger().warn(
                f'Nav2 rejected waypoint #{self.current_waypoint_idx}. Skipping.'
            )
            self.current_waypoint_idx += 1
            self._send_next_nav2_goal()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav2_result)

    def _on_nav2_result(self, future):
        status = None
        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.get_logger().warn(f'Nav2 result callback failed: {exc}')

        if status != 4:
            self.get_logger().warn(
                f'Waypoint #{self.current_waypoint_idx} did not succeed (status={status}).'
            )

        self.current_waypoint_idx += 1
        self._send_next_nav2_goal()

    def _robot_pose_in_map(self):
        if not self.have_odom:
            return None

        if self.map_frame == self.odom_frame:
            return self.odom_x, self.odom_y, self.odom_yaw

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.odom_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
        except TransformException:
            return None

        tx = float(tf_msg.transform.translation.x)
        ty = float(tf_msg.transform.translation.y)
        tq = tf_msg.transform.rotation
        tyaw = yaw_from_quaternion(tq.x, tq.y, tq.z, tq.w)

        cos_yaw = math.cos(tyaw)
        sin_yaw = math.sin(tyaw)
        x_map = tx + cos_yaw * self.odom_x - sin_yaw * self.odom_y
        y_map = ty + sin_yaw * self.odom_x + cos_yaw * self.odom_y
        yaw_map = normalize_angle(self.odom_yaw + tyaw)
        return x_map, y_map, yaw_map

    def _publish_cmd(self, linear, angular):
        if not rclpy.ok():
            return
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        try:
            self.cmd_pub.publish(msg)
        except Exception:
            return

    def _run_cmd_vel_coverage(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            self.stop_execution()
            return

        robot_pose = self._robot_pose_in_map()
        if robot_pose is None:
            self._publish_cmd(0.0, 0.0)
            return

        rx, ry, ryaw = robot_pose
        goal = self.waypoints[self.current_waypoint_idx].pose.position
        dx = float(goal.x) - rx
        dy = float(goal.y) - ry
        distance = math.hypot(dx, dy)

        if distance < self.goal_tolerance_m:
            self.current_waypoint_idx += 1
            self.blocked_since = None
            if self.current_waypoint_idx >= len(self.waypoints):
                self.stop_execution()
            return

        if self.front_distance < self.front_stop_distance:
            now = self.now_sec()
            if self.blocked_since is None:
                self.blocked_since = now

            self._publish_cmd(0.0, 0.4 * self.angular_speed)
            if (now - self.blocked_since) > self.blocked_skip_timeout_sec:
                self.get_logger().warn(
                    f'Skipping blocked waypoint #{self.current_waypoint_idx} due to dynamic obstacle.'
                )
                self.current_waypoint_idx += 1
                self.blocked_since = None
            return

        self.blocked_since = None

        target_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(target_yaw - ryaw)

        if abs(yaw_error) > self.yaw_align_threshold_rad:
            linear = 0.0
            angular = clamp(1.6 * yaw_error, -self.angular_speed, self.angular_speed)
        else:
            linear = clamp(0.6 * distance, 0.03, self.linear_speed)
            angular = clamp(1.2 * yaw_error, -self.angular_speed, self.angular_speed)

        self._publish_cmd(linear, angular)

    def control_loop(self):
        self._maybe_replan()

        if self.exec_mode == 'cmd_vel':
            self._run_cmd_vel_coverage()
            return

        if self.exec_mode in ('idle', 'nav2', 'done'):
            return

    def destroy_node(self):
        try:
            self._publish_cmd(0.0, 0.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CoveragePlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
