from lxros.launch import *

def generate_launch_description():
    return launch(

   group("teleop", 
        #node(
        #    name="teleop_node",
        #    package="haply_ros2",
        #    executable="haply_teleop.py",
        #    remap={"des_vel": "/cartesian_velocity_controller/command",
        #           "robot_force": "/franka_robot_state_broadcaster/external_wrench_in_base_frame",
        #           },),

        node(name=None, package="haply_ros2", executable="haply_node",),
        ),
    #include('franka_lxv', 'franka_lxv_bringup.launch.py')
    )