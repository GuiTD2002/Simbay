def execute_trajectory(robot, trajectory, callback=None):
    """
    Utility: Executes a joint trajectory on Sim or Real hardware.
    """
    for qpos in trajectory:
        robot.move_joints(qpos)

        if callback:
            callback()

        # Let the robot pace itself!
        # -> Sim Robot: Skips rendering if it's going too fast (>60 FPS) to save CPU.
        # -> Real Robot: Calculates exact math time and sleeps the remainder to lock at 1000 Hz.
        robot.sync()

    
    print(f"[DEBUG] Trajectory execution completed")
