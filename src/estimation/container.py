from dataclasses import dataclass
from dataclasses import field

import mujoco

from src.utils import initialize_mujoco_env


@dataclass
class RobotContainer:
    num_particles: int
    props: dict
    dt: float
    
    # These are generated internally, so we tell the dataclass not to ask for them in __init__
    robots: list = field(init=False)
    obj_id: int = field(init=False)
    jnt_adr: int = field(init=False)
    qpos_adr: int = field(init=False)
    dof_adr: int = field(init=False)

    def __post_init__(self):
        """This runs automatically to build the container!"""
        self.robots = [initialize_mujoco_env(self.props, self.dt) for _ in range(self.num_particles)]
        
        base_model = self.robots[0].model
        
        # 1. Get the Body ID
        self.obj_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, "object") # type: ignore       
        
        # 2. Find the Joint associated with this Body
        # (body_jntadr gives the index of the first joint attached to this body)
        self.jnt_adr = base_model.body_jntadr[self.obj_id]
        
        # 3. Find the exact memory address in the qpos array for this Joint
        self.qpos_adr = base_model.jnt_qposadr[self.jnt_adr] 
        
        # 4. Find the exact memory address in the qvel array (degrees of freedom) for this Joint
        self.dof_adr = base_model.jnt_dofadr[self.jnt_adr]
   