import numpy as np


def get_disjoint_trees_no_cycle_3d(size=(10, 10, 10)):
    # two disjoint crosses
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    return crosPair


def get_tiny_loop_with_branches():
    # array of a cycle with branches
    tiny_loop = np.zeros((5, 5), dtype=bool)
    tiny_loop[1:4, 1:4] = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=bool)
    tiny_loop[0, 2] = 1
    tiny_loop[4, 2] = 1
    tiny_loop_with_branches = np.zeros((3, 5, 5), dtype=bool)
    tiny_loop_with_branches[1] = tiny_loop
    return tiny_loop_with_branches


def get_tiny_loops_with_branches():
    # array of 2 cycles with branches
    tiny_loop = np.zeros((10, 10), dtype=bool)
    loop = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=bool)
    tiny_loop[1:4, 1:4] = loop
    tiny_loop[0, 2] = 1
    tiny_loop[4, 2] = 1
    tiny_loop[5:8, 1:4] = loop
    tiny_loop[4, 2] = 1
    tiny_loop[8, 2] = 1
    tiny_loops_with_branches = np.zeros((3, 10, 10), dtype=bool)
    tiny_loops_with_branches[1] = tiny_loop
    return tiny_loops_with_branches


def get_disjoint_crosses(size=(10, 10, 10)):
    # two disjoint crosses
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    return crosPair


def get_single_voxel_line(size=(5, 5, 5)):
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    return sampleLine
