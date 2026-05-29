import numpy as np


def detect_contact(measurements, threshold=0.3):
    """Returns True if the torque magnitude exceeds the threshold."""
    #print(np.linalg.norm(measurements))
    return np.linalg.norm(measurements) > threshold

