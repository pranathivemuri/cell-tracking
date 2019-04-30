import nose.tools
import unittest

import cell_tracking.skeleton_graph_stats as skeleton_stats
import cell_tracking.testlib as testlib

# Straight line in three-space
TEST_PATH_LINE = [(1, 4, 4), (1, 3, 4), (1, 2, 4), (1, 1, 4), (1, 0, 4)]


def _helper_skeleton_stats(arr, cutoff=None):
    # Helper to skeletonize and initialize SkeletonStats class
    if cutoff is None:
        cutoff = 0
    else:
        cutoff = cutoff
    return skeleton_stats.SkeletonStats(arr, cutoff=cutoff)


def test_cutoff():
    single_line_arr = testlib.get_single_voxel_line()
    single_line_stat = _helper_skeleton_stats(
        single_line_arr, cutoff=len(TEST_PATH_LINE))

    expected_stats_segments = [{'branch_points': 0}, {'cycles': 0}]
    expected_obj_results = ['v 1 4 4\n', 'v 1 0 4\n', 'v 1 1 4\n', 'v 1 2 4\n', 'v 1 3 4\n']

    nose.tools.assert_list_equal(
        expected_stats_segments,
        single_line_stat.get_stats_general(single_line_stat.networkx_graph)[0])

    nose.tools.assert_list_equal(
        expected_obj_results,
        single_line_stat.get_stats_general(single_line_stat.networkx_graph)[1])


def test_intersection():
    list1 = [(1, 2, 3), (0, 0, 0), (4, 5, 5)]
    list2 = [(4, 5, 5)]
    nose.tools.assert_equal(
        skeleton_stats.SkeletonStats.intersection(list1, list2),
        list2)


def assert_skeleton_stats(obtained_list, expected_segments, obj_lines, expected_obj_lines):
    # Reduces obtained list of dictionaries of metrics of each segment in the graph to one
    # dictionary and compare with the expected_segments dictionary for equality
    expected_keys = 'nodes', 'length', 'contraction', 'start_point', 'end_point'
    reduced_segments = {key: [] for key in expected_keys}
    for segment in obtained_list:
        if len(segment) == 1:
            for gloabl_attrs, _ in segment.items():
                reduced_segments[gloabl_attrs] = segment[gloabl_attrs]
            continue
        for key in expected_keys:
            reduced_segments[key] += [segment[key]]

    for key, value in expected_segments.items():
        if type(value) is list:
            assert sorted(value) == sorted(reduced_segments[key])
        else:
            assert value == reduced_segments[key]

    nose.tools.assert_equal(len(obj_lines), len(expected_obj_lines))


class SkeletonStatsTestsLine(unittest.TestCase):

    def setUp(self):
        # Initialize SkeletonStats for a single line
        single_line_arr = testlib.get_single_voxel_line()
        self.single_line_stat = _helper_skeleton_stats(single_line_arr)
        self.single_line_nx_g = self.single_line_stat.networkx_graph
        self.expected_segments = [{
            "nodes": 5,
            "length": 4,
            "contraction": 1,
            "start_point": TEST_PATH_LINE[-1],
            "end_point": TEST_PATH_LINE[0],
        }]

    def test_set_vessel_segment_stats(self):
        nose.tools.assert_list_equal(
            self.expected_segments,
            self.single_line_stat._set_vessel_segment_stats(TEST_PATH_LINE))

    def test_single_line_stats(self):
        single_line_stats = self.single_line_stat._single_line_stats(self.single_line_nx_g)
        nose.tools.assert_list_equal(self.expected_segments, single_line_stats[0])
        nose.tools.assert_list_equal([], single_line_stats[1])

    def test_paired_iterator(self):
        nose.tools.assert_list_equal(
            [((1, 4, 4), (1, 3, 4)), ((1, 3, 4), (1, 2, 4)),
             ((1, 2, 4), (1, 1, 4)), ((1, 1, 4), (1, 0, 4))],
            self.single_line_stat._paired_iterator(TEST_PATH_LINE))

    def test_remove_edges_visited_path(self):
        self.single_line_stat._remove_edges_visited_path(self.single_line_nx_g, TEST_PATH_LINE)
        nose.tools.assert_equal(self.single_line_nx_g.number_of_edges(), 0)

    def test_undirected_graph_stats_line(self):
        # straight line graph
        nose.tools.assert_list_equal(self.single_line_stat._undirected_graph_stats(
            self.single_line_nx_g)[0], [])
        nose.tools.assert_list_equal(self.single_line_stat._undirected_graph_stats(
            self.single_line_nx_g)[1], [])


class SkeletonStatsTestsCyclicGraph(unittest.TestCase):

    def setUp(self):
        self.tiny_loop_branch_stat = _helper_skeleton_stats(
            testlib.get_tiny_loop_with_branches())
        self.tiny_loop_branch_nx_g = self.tiny_loop_branch_stat.networkx_graph

    def test_undirected_graph_stats_cyclic_tree(self):
        # Cycle with tree in 3 space
        obtained_list, obj_lines = self.tiny_loop_branch_stat._undirected_graph_stats(
            self.tiny_loop_branch_nx_g)
        expected_segments = {
            'nodes': [5, 2, 2, 5],
            'length': [4, 1, 1, 4],
            'contraction': [0.5, 1, 1, 0.5],
            'start_point': [(1, 1, 2), (1, 3, 2), (1, 0, 2), (1, 1, 2)],
            'end_point': [(1, 3, 2), (1, 4, 2), (1, 1, 2), (1, 3, 2)]
        }

        assert_skeleton_stats(obtained_list, expected_segments, obj_lines, [])


class SkeletonStatsTestsTree(unittest.TestCase):

    def setUp(self):
        self.tree_stat = _helper_skeleton_stats(testlib.get_disjoint_trees_no_cycle_3d())
        self.tree_nx_g = self.tree_stat.networkx_graph
        self.expected_segments = {
            'nodes': [3] * 8,
            'length': [2] * 8,
            'contraction': [1] * 8,
            'start_point': [(5, 7, 7), (5, 7, 5), (5, 7, 7), (5, 5, 7),
                            (0, 2, 0), (0, 0, 2), (0, 2, 2), (0, 2, 2)],
            'end_point': [(5, 7, 9), (5, 7, 7), (5, 9, 7), (5, 7, 7),
                          (0, 2, 2), (0, 2, 2), (0, 4, 2), (0, 2, 4)],
        }

    def test_undirected_graph_stats_acyclic_graph(self):
        # Acyclic graph
        obtained_list, obj_lines = self.tree_stat._undirected_graph_stats(self.tree_nx_g)
        assert_skeleton_stats(obtained_list, self.expected_segments, obj_lines, [])

    def test_branch_to_end_stats(self):
        obtained_list, obj_lines = self.tree_stat._branch_to_end_stats(self.tree_nx_g)
        assert_skeleton_stats(obtained_list, self.expected_segments, obj_lines, [])

    def test_branch_to_branch(self):
        obtained_list, obj_lines = self.tree_stat._branch_to_branch_stats(self.tree_nx_g)
        expected_segments = {
            'nodes': [],
            'length': [],
            'contraction': [],
            'start_point': [],
            'end_point': [],
        }

        assert_skeleton_stats(obtained_list, expected_segments, obj_lines, [])

    def test_get_stats_tree(self):
        branch_nodes, end_nodes = self.tree_stat._branch_and_end_nodes(self.tree_nx_g)
        list_of_perms = list(skeleton_stats.itertools.product(branch_nodes, end_nodes))
        obtained_list, obj_lines = self.tree_stat._get_stats_tree(
            self.tree_nx_g, list_of_perms, branch_nodes, 1)

        assert_skeleton_stats(obtained_list, self.expected_segments, obj_lines, [])

    def test_branch_and_end_nodes(self):
        obtained_branch_nodes, obtained_end_nodes = self.tree_stat._branch_and_end_nodes(
            self.tree_nx_g)
        expected_branch_nodes = set([(0, 2, 2), (5, 7, 7)])
        expected_end_nodes = set([(0, 2, 0), (0, 0, 2), (0, 4, 2), (5, 7, 5),
                                  (5, 7, 9), (0, 2, 4), (5, 9, 7), (5, 5, 7)])

        nose.tools.assert_set_equal(set(obtained_branch_nodes), expected_branch_nodes)
        nose.tools.assert_set_equal(set(obtained_end_nodes), expected_end_nodes)


class SkeletonStatsTestsCyclesGraph(unittest.TestCase):
    def setUp(self):
        self.cycles_branches_stat = _helper_skeleton_stats(
            testlib.get_tiny_loops_with_branches())

    def test_get_stats_general_multiple_cycles_graph(self):
        # Graph with 2 cycles and branches from the cycles
        obtained_list, obj_lines = self.cycles_branches_stat.get_stats_general(
            self.cycles_branches_stat.networkx_graph)

        expected_segments = {
            'nodes': [5, 5, 2, 2, 3, 5, 5],
            'length': [4, 4, 1, 1, 2, 4, 4],
            'contraction': [0.5, 0.5, 1, 1, 1, 0.5, 0.5],
            'start_point': [(1, 1, 2), (1, 5, 2), (1, 7, 2), (1, 0, 2),
                            (1, 3, 2), (1, 5, 2), (1, 1, 2)],
            'end_point': [(1, 3, 2), (1, 7, 2), (1, 8, 2), (1, 1, 2),
                          (1, 5, 2), (1, 7, 2), (1, 3, 2)],
            'branch_points': 4,
            'cycles': 2,
        }
        assert_skeleton_stats(obtained_list, expected_segments, obj_lines, [0] * 26)

    def test_undirected_graph_stats_multiple_cycles_graph(self):
        # Graph with 2 cycles and branches from the cycles
        obtained_list, obj_lines = self.cycles_branches_stat._undirected_graph_stats(
            self.cycles_branches_stat.networkx_graph)
        expected_segments = {
            'nodes': [5, 5, 2, 2, 3, 5, 5],
            'length': [4, 4, 1, 1, 2, 4, 4],
            'contraction': [0.5, 0.5, 1, 1, 1, 0.5, 0.5],
            'start_point': [(1, 5, 2), (1, 1, 2), (1, 7, 2),
                            (1, 0, 2), (1, 3, 2), (1, 1, 2), (1, 5, 2)],
            'end_point': [(1, 7, 2), (1, 3, 2), (1, 8, 2),
                          (1, 1, 2), (1, 5, 2), (1, 3, 2), (1, 7, 2)],
        }
        assert_skeleton_stats(obtained_list, expected_segments, obj_lines, [])
