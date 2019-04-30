import itertools
import time

import networkx as nx

import cell_tracking.networkx_graph_from_array as networkx_graph_from_array
import cell_tracking.vessel_segment_stats as vessel_stats

"""
Find the statistics of a networkx graphical representation of skeleton
node = every voxel on skeleton graph is a node
edge = path between two consecutive nodes
degree = number of edges connected to a node
branch node = node with degree greater than 2
end node = node with degree equal to 1
path = shortest path (path with less nodes and length) between any two nodes
segment = any path between two branch nodes, or two end nodes or branch and an end node
"""


class SkeletonStats:
    """
    Find statistics on a networkx graph of a skeleton where every voxel in a 3D skeleton is a node

    :param arr - 3D Skeleton array
    :param voxel_size - tuple
        param representing a tuple of the voxel size in x, y, and z in um
    :param arr_lower_limits: lower limits for array
    :param arr_upper_limits: upper limits for array
    :param cutoff: int defaults to zero
        branch to end segments of length cutoff are removed if cutoff is nonzero

    :return stats - list
        list of dicts for all segments of the skeleton in the cube
    :return obj_lines - list
        list of strings that can be used to construct an obj file
        containing nodes and all the segment paths in the graph
        writing strings of following each separated by a new line
        v followed by x, y, z coordinates
        l followed by indexes to the nodes that form the segment path
    """
    def __init__(self, arr, arr_lower_limits=None, arr_upper_limits=None, voxel_size=None, cutoff: int=0, debug=False):

        networkx_graph = networkx_graph_from_array.get_networkx_graph_from_array(
            arr,
            arr_lower_limits,
            arr_upper_limits)

        dimensions = len(arr.shape)
        if dimensions == 2:
            voxel_size = (1, 1)
        elif dimensions == 3:
            voxel_size = (1, 1, 1)

        self.networkx_graph = nx.Graph(networkx_graph)
        self.vox_dim = voxel_size
        self.cutoff = cutoff
        self.correct_branch_nodes = []
        self.debug = debug
        if self.debug:
            print("cutoff is {}".format(cutoff))

    def _set_vessel_segment_stats(self, path: list):
        """
        :param stats - list
            ["length": float, "tortuosity": float,
             "contraction": float, "hausdorff dimension": float]

        :return a stats record for a segment based on path
        """
        vessel_segment = vessel_stats.VesselSegment(path, self.vox_dim)
        return [vessel_segment.get_vessel_stats()]

    def _set_obj_line(self, obj_node_index_map: dict={}, path: list=[]):
        """
        :param obj_node_index_map - dict
               integer indexes to the nodes in the graph
        :param path - list
               nodes in a given segment

        :return a l followed by the path of nodes, instead of 2D or 3D node
                an index representing the node is returned

        Example: l 3 4 5 if there is a path in the graph connecting 3rd, 4th, 5th node
        """
        if obj_node_index_map != {}:
            return ["l " + " ".join(str(obj_node_index_map[node]) for node in path) + "\n"]
        else:
            return []

    def _set_stats_long_segments(self, path: list, obj_node_index_map: dict):
        """
        If length of path is greater than cutoff, returns stats and obj_lines

        :param cutoff: int defaults to zero
                       branch to end segments of length cutoff are removed if cutoff is nonzero

        :return stats - list
                list of dicts for all segments of the skeleton in the cube
        :return obj_lines - list
                list of strings that can be used to construct an obj file
                containing nodes and all the segment paths in the graph
        """
        if len(path) > self.cutoff:
            return (
                self._set_vessel_segment_stats(path),
                self._set_obj_line(obj_node_index_map, path))
        else:
            return ([], [])

    def _single_line_stats(self, graph: nx.Graph, obj_node_index_map: dict={}):
        """
        :param  graph - nx.Graph
                find a starting point, then walk the length from end point to end
                point of the contigous segment
        :param  obj_node_index_map - dict
                integer indexes to the nodes in the graph

        :return a list of stats dict and obj_line representing the path
        """
        start_node, end_node = [node for node in graph.nodes() if graph.degree[node] == 1]

        path = nx.shortest_path(graph, source=start_node, target=end_node)
        self._remove_edges_visited_path(graph, path)
        return self._set_stats_long_segments(path, obj_node_index_map)

    def _paired_iterator(self, path: list):
        # Returns list of node, next_node
        return [(e1, e2) for e1, e2 in zip(path[:-1], path[1:])]

    def _remove_edges_visited_path(self, graph: nx.Graph, path: list):
        # Remove visited edges for a given graph in the path
        edges_visited = self._paired_iterator(path)
        graph.remove_edges_from(edges_visited)

    @staticmethod
    def intersection(path1: list, path2: list) -> list:
        # Returns intersection of lists path1 and path2, another list
        return list(set(path1) & set(path2))

    def _undirected_graph_stats(self, graph: nx.Graph, cycles=None, obj_node_index_map: dict={}):
        """
        Find statistics of a undirected graph of a disjoint graph
        Go through each of the cycles, find branch nodes on the cycle
        and find segments attached to it, then compute statistics
        Go through rest of the tree structure in the graph and set the statistics
        also append all the paths traversed in a list obj_lines
        """
        original_graph = nx.Graph(graph)
        if cycles is None:
            cycles = nx.cycle_basis(graph)
        branch_nodes_entire_subgraph, end_nodes_entire_subgraph = self._branch_and_end_nodes(graph)
        stats = []
        obj_lines = []
        # Transiting the cycles
        for cycle_path in cycles:
            node_degrees = [original_graph.degree([node])[node] for node in cycle_path]
            branch_nodes = [node for (node, degree) in zip(cycle_path, node_degrees) if degree > 2]

            if len(branch_nodes) <= 1:
                cycle_path.insert(0, cycle_path[-1])

                stat, obj_line = self._set_stats_long_segments(
                    cycle_path,
                    obj_node_index_map)
                stats += stat
                obj_lines += obj_line
                self._remove_edges_visited_path(graph, cycle_path)
                self.correct_branch_nodes += self.intersection(cycle_path, branch_nodes)
            else:
                # Transiting a single cycle
                for branch_node_1, branch_node_2 in self._paired_iterator(branch_nodes):
                    if nx.has_path(graph, branch_node_1, branch_node_2):
                        path = nx.shortest_path(graph, source=branch_node_1, target=branch_node_2)
                        stat, obj_line = self._set_stats_long_segments(
                            path,
                            obj_node_index_map)
                        stats += stat
                        obj_lines += obj_line
                        self._remove_edges_visited_path(graph, path)
                        self.correct_branch_nodes += self.intersection(path, branch_nodes)

        stat, obj_line = self._branch_to_end_stats(graph, branch_nodes_entire_subgraph,
                                                   end_nodes_entire_subgraph, obj_node_index_map)

        stats += stat
        obj_lines += obj_line
        # there can also be untraced edges of length less than cutoff, do not
        # calculate stats for them
        if graph.number_of_edges() != 0:
            stat, obj_line = self._branch_to_branch_stats(
                graph, branch_nodes_entire_subgraph, obj_node_index_map)
            stats += stat
            obj_lines += obj_line
        return stats, obj_lines

    def _branch_to_end_stats(self, graph: nx.Graph, branch_nodes=None, end_nodes=None,
                             obj_node_index_map: dict={}):
        """
        Find statistics of a tree, comprising segments betweeen branch and end nodes
        also append all the paths traversed in a list obj_lines
        """
        if branch_nodes is None and end_nodes is None:
            branch_nodes, end_nodes = self._branch_and_end_nodes(graph)
        perm_iter = itertools.product(branch_nodes, end_nodes)
        return self._get_stats_tree(graph, perm_iter, branch_nodes, 1, obj_node_index_map)

    def _branch_to_branch_stats(self, graph: nx.Graph, branch_nodes=None,
                                obj_node_index_map: dict={}):
        """
        Find statistics of untraced edges between two branch nodes
        also append all the paths traversed in a list obj_lines
        """
        if branch_nodes is None:
            branch_nodes, _ = self._branch_and_end_nodes(graph)
        perm_iter = itertools.permutations(branch_nodes, 2)
        return self._get_stats_tree(graph, perm_iter, branch_nodes, 2, obj_node_index_map)

    def _get_stats_tree(self, graph: nx.Graph, perm_iter: iter,
                        branch_nodes: list, intersection: int=1,
                        obj_node_index_map: dict={}):
        """
        Find statistics of segments betweeen branch and end nodes
        or two branch nodes depending on the intersection
        perm_iter is an iterator of tuples with source and target on a segment
        intersection is an int giving the sum of branch nodes on a segment, default=1,
        segments between branch and end nodes
        also append all the paths traversed in a list obj_lines
        """
        stats = []
        obj_lines = []
        for source_on_tree, target_on_tree in perm_iter:
            can_connect = nx.has_path(graph, source_on_tree, target_on_tree)
            if not can_connect and source_on_tree != target_on_tree:
                continue

            path = nx.shortest_path(graph, source=source_on_tree, target=target_on_tree)
            # statistics of branch to end point segments of length less than or equal to cutoff
            # are not counted
            # assumption that they are spurious branches due to noise or abrupt, unexpected
            # changes in morphology

            branch_to_branch_segment = intersection == 2
            branch_to_end_segment = intersection == 1
            long_enough = len(path) > self.cutoff

            if branch_to_branch_segment or (long_enough and branch_to_end_segment):
                count_branch_nodes_on_path = sum([1 for node in path if node in branch_nodes])
                if count_branch_nodes_on_path == intersection:
                    stats += self._set_vessel_segment_stats(path)
                    obj_lines += self._set_obj_line(obj_node_index_map, path)
                    self._remove_edges_visited_path(graph, path)
                    self.correct_branch_nodes += self.intersection(path, branch_nodes)

        return stats, obj_lines

    def _branch_and_end_nodes(self, graph: nx.Graph):
        # Find branch and end nodes of the graph
        branch_nodes = [node for node in graph.nodes() if graph.degree[node] > 2]
        end_nodes = [node for node in graph.nodes() if graph.degree[node] == 1]
        return branch_nodes, end_nodes

    def get_stats_general(self, graph: nx.Graph):
        """
        1) go through each of the disjoint graphs
        2) decide if it is one of the following
             a) single node
             b) Contigous segment
             c) tree or tree with cycles
        3) And set stats for each subgraph

        :return a list of dicts for each segment (branch to branch, end to end, branch to end)

        :raises an error if there are any untraced edges in a subgraph

        and adds one single dict per graph for number of branch points
         {"nodes": 9,
          "length": 8,
          "tortuosity": np.inf,
          "contraction": 0,
          "hausdorff_dimension": -0.0}, {'branch points': 5}

        also append all the paths traversed in the graph and nodes in the graph to a list obj_lines
        """
        _disjoint_graphs = list(nx.connected_component_subgraphs(graph))
        stats = []
        cycles = []
        obj_node_index_map = {}  # initialize variables for writing string of node
        # v followed by x, y, x coordinates in the obj file
        #  for each of the sorted nodes
        obj_lines = []
        for index, node in enumerate(graph.nodes()):
            # a obj_node_index_map to transform the nodes (x, y, z) to indexes (beginining with 1)
            obj_node_index_map[node] = index + 1
            # add strings of nodes to obj file
            node_line = \
                "v " + " ".join(str(node[i] * self.vox_dim[i]) for i in range(0, len(node))) + "\n"
            obj_lines.append(node_line)

        for nth_subgraph, skeleton_subgraph in enumerate(_disjoint_graphs):
            num_nodes = skeleton_subgraph.number_of_nodes()
            if self.debug:
                print("subgraph contains {} nodes".format(num_nodes))
            # Properties of this subgraph
            iter_time = time.time()
            unique_degrees = set([degree_item[1] for degree_item in nx.degree(skeleton_subgraph)])
            cycles = nx.cycle_basis(skeleton_subgraph)
            # Single node
            if len(skeleton_subgraph.nodes()) <= 1:
                continue

            # Contigous segment (non-branching)
            if set(unique_degrees) == set((1, 2)) or set(unique_degrees) == {1}:
                stat, obj_line = self._single_line_stats(skeleton_subgraph, obj_node_index_map)
                stats += stat
                obj_lines += obj_line

            # Cyclic or acyclic graph
            else:
                stat, obj_line = self._undirected_graph_stats(
                    skeleton_subgraph, cycles, obj_node_index_map)
                stats += stat
                obj_lines += obj_line
            if self.debug:
                print(
                    "Finished iteration %i in %0.2f s" % (nth_subgraph, time.time() - iter_time))
        stats += self.get_branches_cycles(self.networkx_graph)
        return stats, obj_lines

    def get_branches_cycles(self, graph: nx.Graph):
        stats = []
        if graph.number_of_edges() != 0:
            stats += [{'branch_points': len(set(self.correct_branch_nodes))}]
            stats += [{'cycles': len(nx.cycle_basis(graph))}]
        return stats
