import numpy as np
import math

def dot(v1, v2):
	score = np.dot(v1, v2)
	return score if not math.isnan(score) else 0.0

def cosine(v1, v2):
	score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	return score if not math.isnan(score) else 0.0

def jackard(v1, v2):
	score = np.dot(v1, v2) / (np.linalg.norm(v1)**2 + np.linalg.norm(v2)**2 - np.dot(v1, v2))
	return score if not math.isnan(score) else 0.0

def dice(v1, v2):
	score = 2 * np.dot(v1, v2) / (np.linalg.norm(v1)**2 * np.linalg.norm(v2)**2)
	return score if not math.isnan(score) else 0.0

def sim(v1, v2, similarities='cosine'):
	if (similarities == 'cosine'):
		return cosine(v1, v2)
	elif (similarities == 'dice'):
		return dice(v1, v2)
	elif (similarities == 'dot'):
		return dot(v1, v2)
	elif (similarities == 'jackard'):
		return jackard(v1, v2)
