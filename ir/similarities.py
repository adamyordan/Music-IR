import numpy as np
import math

def cosine(v1, v2):
    score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return score if not math.isnan(score) else 0.0
