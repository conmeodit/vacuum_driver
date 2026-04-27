import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vacuum_driver')
    slam_share = get_package_share_directory('slam_toolbox')
    default_slam_params = os.path.join(pkg_share, 'config', 'slam.yaml')
    default_rviz_config = os.path.join(pkg_share, 'rviz', 'mapping.rviz')
    default_urdf = os.path.join(pkg_share, 'urdf', 'vacuum_robot.urdf')

    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params = LaunchConfiguration('slam_params')
    rviz_config = LaunchConfiguration('rviz_config')
    use_autonomy = LaunchConfiguration('use_autonomy')

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
                'publish_lidar_tf': False,
                'scan_topic': '/scan',
                'odom_topic': '/odom',
                'cmd_vel_topic': '/cmd_vel',
                'base_frame_id': 'base_link',
                'odom_frame_id': 'odom',
                'lidar_frame_id': 'laser',
                'max_linear_speed_mps': 0.16,
                'max_angular_speed_radps': 0.85,
                'scan_filter_enabled': True,
                'scan_filter_window': 5,
                'scan_filter_min_valid_neighbors': 3,
                'scan_filter_outlier_threshold_m': 0.12,
                'scan_filter_temporal_alpha': 0.60,
                'scan_filter_temporal_jump_threshold_m': 0.20,
            }
        ],
    )

    with open(default_urdf, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }
        ],
    )

    autonomy_node = Node(
        package='vacuum_driver',
        executable='autonomous_cleaning_node',
        name='autonomous_cleaning_node',
        output='screen',
        condition=IfCondition(use_autonomy),
        parameters=[
            {
                'map_topic': '/map',
                'scan_topic': '/scan',
                'cmd_vel_topic': '/cmd_vel',
                'map_frame': 'map',
                'base_frame': 'base_link',
                'robot_radius_m': 0.26,
                'front_stop_distance_m': 0.34,
                'emergency_stop_distance_m': 0.22,
                'side_clearance_m': 0.23,
                'max_linear_speed': 0.12,
                'min_linear_speed': 0.035,
                'max_angular_speed': 0.80,
                'frontier_min_cluster_size': 8,
                'frontier_min_distance_m': 0.30,
                'map_stable_duration_sec': 8.0,
                'exploration_settle_sec': 5.0,
                'coverage_spacing_m': 0.24,
                'coverage_visited_radius_m': 0.24,
                'coverage_required_ratio': 0.985,
                'stuck_timeout_sec': 5.0,
                'stuck_min_progress_m': 0.08,
            }
        ],
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_share, 'launch', 'online_async_launch.py')
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
            DeclareLaunchArgument('use_autonomy', default_value='true'),
            DeclareLaunchArgument('slam_params', default_value=default_slam_params),
            DeclareLaunchArgument('rviz_config', default_value=default_rviz_config),
            driver_node,
            robot_state_publisher_node,
            slam_launch,
            slam_session_manager_node,
            autonomy_node,
            rviz_node,
        ]
    )
