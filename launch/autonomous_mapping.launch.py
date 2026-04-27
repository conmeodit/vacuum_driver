from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('vacuum_driver')
    slam_share = get_package_share_directory('slam_toolbox')
    default_slam_params = f'{pkg_share}/config/slam.yaml'
    default_rviz_config = f'{pkg_share}/rviz/mapping.rviz'

    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params = LaunchConfiguration('slam_params')
    rviz_config = LaunchConfiguration('rviz_config')

    driver_node = Node(
        package='vacuum_driver',
        executable='pure_driver',
        name='pure_webots_driver',
        output='screen',
        parameters=[
            {
                'use_encoder_odom': True,
                'use_open_loop_odom': False,
                'use_imu_for_odom': False,
                'imu_yaw_blend': 0.20,
                'publish_odom_tf': True,
                'publish_lidar_tf': True,
                'scan_topic': '/scan',
                'odom_topic': '/odom',
                'cmd_vel_topic': '/cmd_vel',
                'base_frame_id': 'base_link',
                'odom_frame_id': 'odom',
                'lidar_frame_id': 'laser',
                'max_linear_speed_mps': 0.16,
                'max_angular_speed_radps': 0.8,
                'scan_filter_enabled': True,
                'scan_filter_window': 5,
                'scan_filter_min_valid_neighbors': 3,
                'scan_filter_outlier_threshold_m': 0.12,
                'scan_filter_temporal_alpha': 0.60,
                'scan_filter_temporal_jump_threshold_m': 0.20,
            }
        ],
    )

    exploration_node = Node(
        package='vacuum_driver',
        executable='exploration_node',
        name='exploration_node',
        output='screen',
        parameters=[
            {
                'scan_topic': '/scan',
                'odom_topic': '/odom',
                'cmd_vel_topic': '/cmd_vel',
                'waypoint_topic': '/coverage_waypoints',
                'linear_speed': 0.10,
                'min_linear_speed': 0.05,
                'max_linear_speed': 0.15,
                'angular_speed': 0.60,
                'max_angular_speed': 0.80,
                'backup_speed': 0.06,
                'front_stop_distance': 0.40,
                'emergency_stop_distance': 0.24,
                'align_front_block_distance': 0.32,
                'side_clearance': 0.30,
                'rear_clearance': 0.22,
                'goal_tolerance_m': 0.20,
                'coverage_min_target_distance_m': 0.85,
                'backup_duration_sec': 0.80,
                'turn_min_duration_sec': 0.75,
                'turn_max_duration_sec': 1.60,
                'avoid_duration_sec': 1.20,
                'avoid_clear_hold_sec': 0.70,
                'stuck_timeout_sec': 4.5,
                'stuck_min_progress_m': 0.06,
                'local_loop_timeout_sec': 24.0,
                'local_loop_radius_m': 0.85,
                'local_loop_min_path_m': 1.8,
                'map_topic': '/map',
                'frontier_enabled': True,
                'frontier_search_stride_cells': 1,
                'frontier_clearance_cells': 4,
                'frontier_min_cluster_size': 10,
                'frontier_min_unknown_neighbors': 2,
                'frontier_min_distance_m': 0.90,
                'frontier_replan_interval_sec': 1.5,
                'frontier_target_timeout_sec': 12.0,
                'frontier_target_min_hold_sec': 5.0,
                'frontier_blocked_skip_timeout_sec': 6.0,
                'frontier_blacklist_radius_m': 0.80,
                'frontier_blacklist_timeout_sec': 90.0,
                'planner_forward_weight': 0.35,
                'planner_in_place_penalty': 0.35,
            }
        ],
    )

    coverage_planner_node = Node(
        package='vacuum_driver',
        executable='coverage_planner_node',
        name='coverage_planner_node',
        output='screen',
        parameters=[
            {
                'map_topic': '/map',
                'odom_topic': '/odom',
                'scan_topic': '/scan',
                'waypoint_topic': '/coverage_waypoints',
                'map_frame': 'map',
                'odom_frame': 'odom',
                'grid_stride_cells': 3,
                'obstacle_threshold': 50,
                'unknown_is_obstacle': True,
                'obstacle_inflation_cells': 4,
                'visited_radius_m': 0.19,
                'path_sampling_stride_cells': 3,
                'max_waypoints_per_plan': 80,
                'auto_plan_on_first_map': True,
                'auto_replan': True,
                'replan_period_sec': 6.0,
                'replan_min_robot_shift_m': 0.60,
                'auto_start_execution': False,
                'use_nav2': False,
            }
        ],
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{slam_share}/launch/online_async_launch.py'
        ),
        launch_arguments={
            'slam_params_file': slam_params,
            'use_sim_time': use_sim_time,
            'autostart': 'true',
            'use_lifecycle_manager': 'false',
        }.items(),
    )

    slam_session_manager_node = Node(
        package='vacuum_driver',
        executable='slam_session_manager_node',
        name='slam_session_manager_node',
        output='screen',
        parameters=[
            {
                'startup_delay_sec': 2.0,
                'service_wait_timeout_sec': 20.0,
                'shutdown_after_reset': True,
            }
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='false'),
            DeclareLaunchArgument('use_rviz', default_value='true'),
            DeclareLaunchArgument('slam_params', default_value=default_slam_params),
            DeclareLaunchArgument('rviz_config', default_value=default_rviz_config),
            driver_node,
            exploration_node,
            coverage_planner_node,
            slam_launch,
            slam_session_manager_node,
            rviz_node,
        ]
    )
