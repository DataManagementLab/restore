import numpy as np
from numba import njit, prange, boolean


@njit()
def in_set(matrix, index_to_keep):
    out = np.empty(matrix.shape[0], dtype=boolean)
    index_to_keep_set = set(index_to_keep)
    matrix = matrix.flatten()

    for i in prange(matrix.shape[0]):
        if matrix[i] in index_to_keep_set:
            out[i] = True
        else:
            out[i] = False

    return out
