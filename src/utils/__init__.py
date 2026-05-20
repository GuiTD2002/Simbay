"""
Utilities module providing simulation factories, model loading tools, 
and hardware-agnostic physics wrappers for MuJoCo and PyBullet.
"""

from .constants import DEFAULT_OBJECT_PROPS
from .constants import DEFAULT_OBJECT_PROPS2
from .constants import FRANKA_HOME_QPOS
from .constants import FRANKA_HOME_XYZ
from .mujoco_utils import initialize_mujoco_env
from .mujoco_utils import load_mujoco_model
from .mujoco_utils import modify_object_properties
from .sensors import detect_contact
from .visuals import visualize_particles

from.visuals import plot_particle_evolution

from. execution import execute_trajectory

# Defines the public API for the utilities module
__all__ = [
    "DEFAULT_OBJECT_PROPS",
    "DEFAULT_OBJECT_PROPS2",
    "FRANKA_HOME_QPOS",
    "FRANKA_HOME_XYZ",
    "initialize_mujoco_env",
    "load_mujoco_model",
    "modify_object_properties",
    "detect_contact",
    "visualize_particles",
    "execute_trajectory",
    "plot_particle_evolution"
]
