from abc import ABC
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseRobot(ABC):
    """
    Abstract Interface that both Sim and Real robots must implement.
    """
    
    @abstractmethod
    def move_joints(self, pos: NDArray[np.float64]) -> None:
        """
        Moves the robot to the target joint configuration.
        """
        ...
    
    def move_gripper(self, width: float) -> None:
        """ 
        Moves the robot gripper to the target width.
        """
        ...

    @abstractmethod
    def get_joints_pos(self) -> NDArray[np.float64]:
        """
        Returns the current joint positions.
        """
        ...

    @abstractmethod
    def get_joints_vel(self) -> NDArray[np.float64]:
        """
        Returns the current joint velocities.
        """
        ...

    @abstractmethod
    def get_ee_pos(self) -> NDArray[np.float64]:
        """
        Returns the current end-effector position.
        """
        ...

    @abstractmethod
    def get_torque_reads(self) -> NDArray[np.float64]:
        """
        Returns sensory torque reads.
        """
        ...

    @abstractmethod
    def get_force_reads(self) -> NDArray[np.float64]:
        """
        Returns sensory force reads.
        """
        ...
