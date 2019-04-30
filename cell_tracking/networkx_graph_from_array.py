import itertools

import numpy as np
import networkx as nx

from scipy import ndimage


"""
program to look up adjacent elements and calculate degree
this dictionary can be used for graph creation
since networkx graph based on looking up the array and the
adjacent coordinates takes long time. create a dict
using dict_of_indices_and_adjacent_coordinates.
Following are the 27 position vectors of 3 x 3 x 3 second ordered neighborhood of a voxel
at origin (0, 0, 0)
(-1 -1 -1) (-1 0 -1) (-1 1 -1)
(-1 -1 0)  (-1 0 0)  (-1 1 0)
(-1 -1 1)  (-1 0 1)  (-1 1 1)
(0 -1 -1) (0 0 -1) (0 1 -1)
(0 -1 0)  (0 0 0)  (0 1 0)
(0 -1 1)  (0 0 1)  (0 1 1)
(1 -1 -1) (1 0 -1) (1 1 -1)
(1 -1 0)  (1 0 0)  (1 1 0)
(1 -1 1)  (1 0 1)  (1 1 1)
"""
TEMPLATE_3D = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                        [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                        [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)

TEMPLATE_2D = np.array([[2 ** 0, 2 ** 1, 2 ** 2], [2 ** 3, 0, 2 ** 4], [2 ** 5, 2 ** 6, 2 ** 7]])

# permutations of (-1, 0, 1) in three/two dimensional tuple format
# representing 8 and 26 increments around a pixel at origin (0, 0, 0)
# 2nd ordered neighborhood around a voxel/pixel
LIST_POSITION_VECTORS3D = list(itertools.product((-1, 0, 1), repeat=3))
LIST_POSITION_VECTORS3D.remove((0, 0, 0))

LIST_POSITION_VECTORS2D = list(itertools.product((-1, 0, 1), repeat=2))
LIST_POSITION_VECTORS2D.remove((0, 0))


def _get_position_vectors(config_number, dimensions):
    """
    Return a list of tuples of position vectors
    If dimensions are not 2 or 3, raises an assertion error
    Parameters
    ----------
    config_number : int64
        integer less than 2 ** 26
    dimensions: int
        number of dimensions, can only be 2 or 3
    Returns
    -------
    list
        a list of position vectors of a non zero voxel/pixel
        if it is a zero voxel, an empty tuple is returned, else
        the position vector of non zero voxel is returned
    Notes
    ------
    As in the beginning of the program, there are position vectors
    around a voxel at origin (0, 0, 0) which are returned by this function.
    config_number is a decimal number representation of 26 binary numbers
    around a voxel at the origin in a second ordered neighborhood
    """
    config_number = np.int64(config_number)
    if dimensions == 3:
        neighbor_values = [(config_number >> digit) & 0x01 for digit in range(26)]
        position_vectors = LIST_POSITION_VECTORS3D
    elif dimensions == 2:
        neighbor_values = [(config_number >> digit) & 0x01 for digit in range(8)]
        position_vectors = LIST_POSITION_VECTORS2D
    return [neighbor_value * position_vector for neighbor_value, position_vector in zip(neighbor_values, position_vectors)]


def set_adjacency_list(arr):
    """
    Return dict
    Parameters
    ----------
    arr
    Returns
    -------
    dict_of_indices_and_adjacent_coordinates: Dictionary
        key is the nonzero coordinate
        is all the position of nonzero coordinates around it
        in it's second order neighborhood
    """
    coordinate_bitmask_lists = set_bitmask_lists(arr)
    dict_of_indices_and_adjacent_coordinates = {}
    # list of unique nonzero tuples
    for non_zero, config_number in coordinate_bitmask_lists:
        adjacent_coordinate_list = [tuple(np.add(non_zero, position_vector))
                                    for position_vector in _get_position_vectors(config_number, arr.ndim)
                                    if position_vector != ()]
        dict_of_indices_and_adjacent_coordinates[tuple(non_zero)] = adjacent_coordinate_list
    return dict_of_indices_and_adjacent_coordinates


def set_bitmask_lists(arr):
    dimensions = arr.ndim
    # convert the binary array to a configuration number array of same size by convolving with template
    if dimensions == 3:
        result = ndimage.convolve(np.uint64(arr), TEMPLATE_3D, mode='constant')
    elif dimensions == 2:
        result = ndimage.convolve(np.uint64(arr), TEMPLATE_2D, mode='constant')
    non_zero_coordinates = list(set(map(tuple, np.transpose(np.nonzero(arr)))))
    return [[[int(posn) for posn in non_zero], int(result[non_zero])] for non_zero in non_zero_coordinates]


def _get_cliques_of_size(networkx_graph, clique_size):
    """
    Return cliques of size "clique_size" in networkx_graph
    Parameters
    ----------
    networkx_graph : Networkx graph
        graph to obtain cliques from
    Returns
    -------
        list
        list of edges forming 3 vertex cliques
    """
    cliques = nx.find_cliques_recursive(networkx_graph)
    # all the nodes/vertices of 3 cliques
    return [clique for clique in cliques if len(clique) == clique_size]


def _reduce_clique(clique_edges: list, combination_edges: list,
                   mth_clique: int, mth_clique_edge_length_list: list):
    """
    the edge with maximum edge length in case of a right
    angled clique (1, 1, sqrt(2)) or with distances (sqrt(2), sqrt(2), sqrt(2))
    Note -- here squared distance is checked
    """
    for nth_edge_in_mth_clique, edge_length in enumerate(mth_clique_edge_length_list):
        case_a_clique = (
            edge_length == 2 and np.unique(mth_clique_edge_length_list).tolist() == [1, 2])
        case_b_clique = edge_length == 2 and np.unique(mth_clique_edge_length_list).tolist() == [2]
        if case_a_clique:
            clique_edges.append(combination_edges[mth_clique][nth_edge_in_mth_clique])
        if case_b_clique:
            clique_edges.append(combination_edges[mth_clique][nth_edge_in_mth_clique])
            break


def _remove_clique_edges(networkx_graph: nx.Graph):
    """
    Return 3 vertex clique removed networkx graph changed in place
    :param networkx_graph: Networkx graph to remove cliques from
    :return networkx_graph: Networkx graph changed in place
            graph with 3 vertex clique edges removed
    Notes
    ------
    Returns networkx graph changed in place after removing 3 vertex cliques
    Raises AssertionErrorr if number of graphs after clique removal has changed
    """
    three_vertex_cliques = _get_cliques_of_size(networkx_graph, clique_size=3)
    combination_edges = [list(itertools.combinations(clique, 2)) for clique in three_vertex_cliques]
    # clique_edge_lengths is a list of lists, where each list is
    # the length of an edge in 3 vertex clique
    clique_edge_lengths = []
    # different combination of edges are in combination_edges and their corresponding lengths are in
    # clique_edge_lengths
    for combination_edge in combination_edges:
        clique_edge_lengths.append([np.sum((np.array(edge[0]) - np.array(edge[1])) ** 2)
                                    for edge in combination_edge])
    # clique edges to be removed are collected here in the for loop below
    clique_edges = []
    for mth_clique, mth_clique_edge_length_list in enumerate(clique_edge_lengths):
        _reduce_clique(clique_edges, combination_edges, mth_clique, mth_clique_edge_length_list)

    networkx_graph.remove_edges_from(clique_edges)


def get_networkx_graph_from_array(arr, arr_lower_limits=None, arr_upper_limits=None):
    """
    Return a networkx graph
    Parameters
    ----------
    :param arr: 3D skeleton array
    :param arr_lower_limits: lower limits for array
    :param arr_upper_limits: upper limits for array
    Returns
    -------
    networkx_graph : Networkx graph
        graphical representation of the input array after clique removal
    """
    shape = arr.shape
    if arr_lower_limits is None:
        if len(shape) == 3:
            arr_lower_limits = (0, 0, 0)
        elif len(shape) == 2:
            arr_lower_limits = (0, 0)
    if arr_upper_limits is None:
        arr_upper_limits = shape
    networkx_graph = nx.from_dict_of_lists(set_adjacency_list(arr))
    _remove_clique_edges(networkx_graph)
    return networkx_graph
