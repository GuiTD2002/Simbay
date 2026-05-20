# ==========================================
# 1. IMPORTS
# ==========================================
import os

import mujoco
import numpy as np

from src.robots import MujocoRobot

from .constants import DEFAULT_OBJECT_PROPS
from .constants import FRANKA_HOME_QPOS


# ==========================================
# 2. CORE MUJOCO LOADER
# ==========================================
def load_mujoco_model(xml_path):
    """
    Reads the XML from the path and builds a fresh, independent C++ model.
    The OS caches this file in RAM automatically, making it safe and lightning-fast
    to call this 100 times in a loop for the Particle Filter.
    """
    try:
        model = mujoco.MjModel.from_xml_path(xml_path) # type: ignore
        data = mujoco.MjData(model) # type: ignore
    except ValueError as e:
        raise ValueError(f"Error loading MuJoCo XML from {xml_path}: {e}")
    
    # Configure initial state
    data.qpos[:7] = FRANKA_HOME_QPOS
    data.ctrl[:7] = FRANKA_HOME_QPOS
    mujoco.mj_forward(model, data) # type: ignore

    return model, data


# ==========================================
# 3. PHYSICS MODIFIER
# ==========================================
def modify_object_properties(model, data, body_name, props):
    """
    Modifies a MuJoCo body safely, preserving the robot's state and 
    strictly trusting the XML for contact physics unless explicitly overridden.
    """
    try:
        body_id = model.body(body_name).id
        geom_start = model.body_geomadr[body_id]
        geom_num = model.body_geomnum[body_id]
    except KeyError:
        print(f"ERROR: Body '{body_name}' not found!")
        return

    # ==========================================
    # 1. THE STATE SHIELD (Stops the Robot Snapping)
    # ==========================================
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()

    # ==========================================
    # 2. MODIFY STRUCTURE (Preserving XML Physics)
    # ==========================================
    for i in range(geom_num):
        geom_id = geom_start + i
        
        if "type" in props:
            type_map = {
                "box":      mujoco.mjtGeom.mjGEOM_BOX,      # type: ignore
                "sphere":   mujoco.mjtGeom.mjGEOM_SPHERE,   # type: ignore
                "capsule":  mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, # type: ignore
            }
            if props["type"] in type_map:
                model.geom_type[geom_id] = type_map[props["type"]]
                
        if "size" in props:
            model.geom_size[geom_id] = np.array(props["size"])
            
        if "friction" in props:
            model.geom_friction[geom_id] = np.array(props["friction"])
        if "solref" in props:
            model.geom_solref[geom_id] = np.array(props["solref"])
        if "solimp" in props:
            model.geom_solimp[geom_id] = np.array(props["solimp"])

    if "mass" in props:
        model.body_mass[body_id] = props["mass"]

    mujoco.mj_setConst(model, data) # type: ignore

    # ==========================================
    # 3. RESTORE THE SHIELD
    # ==========================================
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.ctrl[:] = saved_ctrl

    # ==========================================
    # 4. APPLY DYNAMIC POSITION & ROTATION
    # ==========================================
    if "pos" in props or "angle" in props:
        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr != -1:
            qpos_adr = model.jnt_qposadr[jnt_adr]
            dof_adr = model.jnt_dofadr[jnt_adr]
            
            # A. Move the block
            if "pos" in props:
                data.qpos[qpos_adr : qpos_adr+3] = np.array(props["pos"])
                
            # B. Rotate the block using MuJoCo's native engine
            if "angle" in props:
                # MuJoCo reads Euler angles as [Roll, Pitch, Yaw(Z)]
                euler = np.array([0.0, 0.0, props["angle"]]) 
                quat = np.zeros(4)
                mujoco.mju_euler2Quat(quat, euler, "XYZ") # type: ignore
                data.qpos[qpos_adr+3 : qpos_adr+7] = quat
            
            # C. Zero out object velocity so it doesn't drift
            data.qvel[dof_adr : dof_adr+6] = 0.0

    # Apply changes instantly
    mujoco.mj_forward(model, data) # type: ignore


# ==========================================
# 4. FACTORY
# ==========================================
def initialize_mujoco_env(object_properties=DEFAULT_OBJECT_PROPS, dt=0.002):
    xml_path = os.path.join("models", "scene.xml")
    
    model, data = load_mujoco_model(xml_path)
    modify_object_properties(model, data, "object", object_properties)

    
    return MujocoRobot(model, data, dt=dt)
