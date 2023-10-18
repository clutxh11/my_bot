from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Set the path to your canny edge detection Python script
    #canny_script_path = os.path.join(os.getcwd(), 'src', 'my_bot', 'my_bot', 'canny_node.py')

    # Define launch parameters (if needed)
    #low_threshold = LaunchConfiguration('low_threshold', default='0')
    #high_threshold = LaunchConfiguration('high_threshold', default='50')

    canny = Node(
        package="my_bot",
        executable="canny_node",
        name="canny"
    )

    return LaunchDescription([
        # Add actions here, such as starting your Canny node
        #ExecuteProcess(
        #   cmd=['python3', canny_script_path],
        #   output='screen',
        #   arguments=[low_threshold, high_threshold],
        #),
        # You can add more actions here for other nodes, parameters, etc.
        canny
    ])
