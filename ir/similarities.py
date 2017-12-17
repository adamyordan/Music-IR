import numpy as np

def cosine(v1, v2):
    return float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
