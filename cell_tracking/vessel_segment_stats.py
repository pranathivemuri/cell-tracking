import numpy as np


class VesselSegment:
    """Given a path (list of nodes) and voxel size compute:
     - Count
     - Length
     - Tortuosity
     - Contraction
     - Hausdorff Dimension

    :param voxel_size - param representing a tuple of the voxel size in x, y, and z
    """
    def __init__(self, path: list, voxel_size: tuple=None):
        self.path = path
        if voxel_size is None:
            dimensions = len(path[0])
            if dimensions == 3:
                voxel_size = (1, 1, 1)
            elif dimensions == 2:
                voxel_size = (1, 1)
        self.voxel_size = voxel_size

    def _dist_between_nodes(self, node1, node2):
        vect = np.array(node1, dtype=np.float64) - np.array(node2, dtype=np.float64)
        vect = np.multiply(vect, self.voxel_size)
        return np.linalg.norm(vect)

    def get_displacement(self):
        return self._dist_between_nodes(self.path[0], self.path[-1])

    def get_length(self):
        dist = 0
        for node1, node2 in zip(self.path[:-1], self.path[1:]):
            dist += self._dist_between_nodes(node1, node2)
        return dist

    def get_tortuosity(self):
        curve_length = self.get_length()
        displacement = self.get_displacement()

        return curve_length / displacement

    def get_contraction(self):
        return 1.0 / self.get_tortuosity()

    def get_hausdorff_dimension(self):
        displacement = self.get_displacement()
        length = self.get_length()
        if length == displacement:
            return 1.0
        return np.log(length) / np.log(displacement)

    def _cast_to_int(self, node):
        return tuple([int(i) for i in node])

    def get_vessel_stats(self):
        if np.linalg.norm(self.path[0]) > np.linalg.norm(self.path[-1]):
            start_point, end_point = self.path[-1], self.path[0]
        else:
            start_point, end_point = self.path[0], self.path[-1]
        return {
            "nodes": len(self.path),
            "length": self.get_length(),
            "contraction": self.get_contraction(),
            "start_point": self._cast_to_int(start_point),
            "end_point": self._cast_to_int(end_point)
        }
