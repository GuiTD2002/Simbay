"""
Global constants and default physics properties for the Simbay project.
"""

import numpy as np

# Default properties for objects spawned in simulation
DEFAULT_OBJECT_PROPS = {
    #"type":     "box",   
    #"size":     (0.25/2, 0.15/2, 0.08/2),      
    "mass":     10000.0,                    
    "pos":      (0.55, 0.15, 0.07),
    "angle": np.radians(0)                     
}

FRANKA_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

FRANKA_HOME_XYZ = np.array([0.55450, -0.00000, 0.52109])

#x = 25 cm
#y = 15 cm
#Z = 8 cm

DEFAULT_OBJECT_PROPS2 = {
    "type":     "box",   
    "size":     (0.03, 0.03, 0.03),      
    "mass":     0.100,                    
    "pos":      (0.55, 0.15, 0.031),
    "angle": np.radians(0)                     
}
