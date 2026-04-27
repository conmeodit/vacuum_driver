from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('vacuum_driver')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    default_params = f'{pkg_share}/config/nav2_params.yaml'
    default_rviz_config = f'{pkg_share}/rviz/mapping.rviz'

    map_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_config = LaunchConfiguration('rviz_config')

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{nav2_bringup_share}/launch/navigation_launch.py'
        ),
        launch_arguments={
            'map': map_file,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
        }.items(),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_nav2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('map', description='Absolute path to map yaml file'),
            DeclareLaunchArgument('use_sim_time', default_value='false'),
            DeclareLaunchArgument('params_file', default_value=default_params),
            DeclareLaunchArgument('use_rviz', default_value='true'),
            DeclareLaunchArgument('rviz_config', default_value=default_rviz_config),
            nav2_launch,
            rviz_node,
        ]
    )
