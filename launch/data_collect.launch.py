import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_name = 'imitation_nav_training'

    config_path = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'params.yaml'
    )

    with open(config_path, 'r') as file:
        launch_params = yaml.safe_load(file)['launch']['ros__parameters']

    # データ収集ノード
    data_collect_node = Node(
        package=pkg_name,
        executable='data_collector_node',
        name='data_collector',
        output='screen',
        parameters=[config_path]
    )

    image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_viewer',
        output='screen',
        arguments=['--force-discover']
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    launch_discrption = LaunchDescription()

    if(launch_params['image_view'] is True):
        launch_discrption.add_entity(image_view)
    if(launch_params['rviz'] is True):
        launch_discrption.add_entity(rviz)

    launch_discrption.add_entity(data_collect_node)

    return launch_discrption