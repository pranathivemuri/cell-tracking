import networkx as nx
import nose.tools
import numpy as np

import cell_tracking.networkx_graph_from_array as networkx_graph_from_array
import cell_tracking.testlib as testlib


def _helper_networkx_graph(networkx_graph_array, expected_edges):
    networkx_graph = networkx_graph_from_array.get_networkx_graph_from_array(networkx_graph_array)
    obtained_edges = networkx_graph.number_of_edges()
    nose.tools.assert_greater_equal(obtained_edges, expected_edges)


def test_tiny_loop_with_branches():
    # a loop and a branches coming at end of the cycle
    _helper_networkx_graph(testlib.get_tiny_loop_with_branches(), 10)


def test_disjoint_crosses():
    # two disjoint crosses
    _helper_networkx_graph(testlib.get_disjoint_trees_no_cycle_3d(), 16)


def test_single_voxel_line():
    _helper_networkx_graph(testlib.get_single_voxel_line(), 4)


def test_get_position_vectors():
    expected_cases = [[1, 0], [1, 0], [2**25, 25]]
    for input_args, result in expected_cases:
        nose.tools.assert_true(networkx_graph_from_array._get_position_vectors(input_args, 3)[result])


def test_get_cliques_of_size():
    networkx_graph_with_cliques = nx.diamond_graph()
    nose.tools.assert_equal(len(networkx_graph_from_array._get_cliques_of_size(
        networkx_graph_with_cliques, 3)), 2)


def test_remove_clique_edges():
    networkx_graph_with_cliques = nx.diamond_graph()
    edges_before = networkx_graph_with_cliques.number_of_edges()
    networkx_graph_from_array._remove_clique_edges(networkx_graph_with_cliques)
    nose.tools.assert_equal(networkx_graph_with_cliques.number_of_edges(), edges_before)


def test_do_not_remove_clique_edges_not_one_pixel_apart():
    networkx_graph_with_cliques = nx.Graph()
    edges = [(0, 3), (0, 4), (3, 4)]
    edges_before = len(edges)
    networkx_graph_with_cliques.add_edges_from(edges)
    networkx_graph_from_array._remove_clique_edges(networkx_graph_with_cliques)
    nose.tools.assert_equal(networkx_graph_with_cliques.number_of_edges(), edges_before)


def test_remove_clique_edges_one_pixel_apart():
    networkx_graph_with_cliques = nx.Graph()
    edges = [((2, 2, 2), (2, 2, 1)), ((2, 2, 2), (2, 1, 2)), ((2, 2, 1), (2, 1, 2))]
    networkx_graph_with_cliques.add_edges_from(edges)
    networkx_graph_from_array._remove_clique_edges(networkx_graph_with_cliques)
    nose.tools.assert_equal(networkx_graph_with_cliques.number_of_edges(), 2)


def test_remove_clique_edges_sqrt2_pixel_apart():
    networkx_graph_with_cliques = nx.Graph()
    edges = [
        ((0, 0, 0), (1, 0, 1)), ((0, 0, 0), (1, 1, 0)), ((1, 0, 1), (1, 1, 0)),
        ((3, 3, 3), (4, 3, 4)), ((3, 3, 3), (4, 4, 3)), ((4, 3, 4), (4, 4, 3))]
    networkx_graph_with_cliques.add_edges_from(edges)
    networkx_graph_from_array._remove_clique_edges(networkx_graph_with_cliques)
    nose.tools.assert_equal(networkx_graph_with_cliques.number_of_edges(), 4)


def test_empty_array():
    _helper_networkx_graph(np.zeros((10, 10, 1), dtype=bool), 0)


def test_all_one_array():
    _helper_networkx_graph(np.ones((10, 10, 1), dtype=bool), 0)
