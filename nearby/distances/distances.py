import numpy as np


def distance_angle(w1, w2):
    return np.arccos(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2)))
