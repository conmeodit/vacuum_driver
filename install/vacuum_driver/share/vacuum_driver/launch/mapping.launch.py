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
                'scan_filter_enabled': True,
                'scan_filter_window': 5,
                'scan_filter_min_valid_neighbors': 3,
                'scan_filter_outlier_threshold_m': 0.12,
                'scan_filter_temporal_alpha': 0.60,
                'scan_filter_temporal_jump_threshold_m': 0.20,
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
            slam_launch,
            slam_session_manager_node,
            rviz_node,
        ]
    )
