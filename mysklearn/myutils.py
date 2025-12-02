"""
utility functions for mysklearn package
"""

import numpy as np

def my_euclidean_distance(point1, point2):
    """compute euclidean distance"""
    diff = np.array(point1) - np.array(point2)
    return float(np.sqrt(np.sum(diff ** 2)))
