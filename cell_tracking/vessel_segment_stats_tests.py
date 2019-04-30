import nose.tools
import unittest

import numpy as np

import cell_tracking.vessel_segment_stats as vessel_stats


# Straight line in three-space
TEST_PATH_LINE = [(1, 4, 4), (1, 3, 4), (1, 2, 4), (1, 1, 4), (1, 0, 4)]
TEST_PATH_CYCLE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]


class VesselStatsTests(unittest.TestCase):
    def setUp(self):
        self.voxel_sizes = [(1, 1, 1), (1, 2, 0.5), (1, 3, 5), (0.7, 0.7, 5)]
        self.expected_lengths_line = [4, 8, 12, 2.8]
        self.expected_lengths_cycle = [4, 6, 8, 2.8]

    def test_vesselsegment_get_displacement(self):
        for voxel_size, displacement in zip(self.voxel_sizes, self.expected_lengths_line):
            vs_line = vessel_stats.VesselSegment(TEST_PATH_LINE, voxel_size)
            nose.tools.assert_equal(vs_line.get_displacement(), displacement)
            vs = vessel_stats.VesselSegment([
                (1, 1, 1),
                (2, 2, 2),
                (1, 1, 1),
            ], voxel_size)
            nose.tools.assert_equal(vs.get_displacement(), 0)

    def test_vesselsegment_get_length(self):
        for voxel_size, length_line, length_cycle in zip(
                self.voxel_sizes,
                self.expected_lengths_line, self.expected_lengths_cycle):
            vs_line = vessel_stats.VesselSegment(TEST_PATH_LINE, voxel_size)
            nose.tools.assert_equal(vs_line.get_length(), length_line)

            vs = vessel_stats.VesselSegment([
                (0, 0, 0),
                (2, 3, 4),
            ], voxel_size)
            dist = np.sqrt((2 * voxel_size[0]) ** 2 +
                           (3 * voxel_size[1]) ** 2 +
                           (4 * voxel_size[2]) ** 2)
            nose.tools.assert_equal(vs.get_length(), dist)

            vs = vessel_stats.VesselSegment(TEST_PATH_CYCLE, voxel_size)
            print(voxel_size, vs.get_length(), length_cycle)
            nose.tools.assert_equal(vs.get_length(), length_cycle)

    def test_vesselsegment_tortuosity_and_contraction(self):
        for voxel_size in self.voxel_sizes:
            vs_line = vessel_stats.VesselSegment(TEST_PATH_LINE, voxel_size)
            nose.tools.assert_equal(vs_line.get_tortuosity(), 1)
            nose.tools.assert_equal(vs_line.get_tortuosity(), 1 / vs_line.get_contraction())

            # Test the limit behaviors
            vs = vessel_stats.VesselSegment(TEST_PATH_CYCLE, voxel_size)
            nose.tools.assert_equal(vs.get_tortuosity(), np.inf)

            vs = vessel_stats.VesselSegment(TEST_PATH_CYCLE, voxel_size)
            nose.tools.assert_equal(vs.get_contraction(), 0)
        # TODO(meawoppl) this deserves a more sophisticated test.

    def test_vesselsegment_get_hausdorff_dimension(self):
        for voxel_size in self.voxel_sizes:
            vs_line = vessel_stats.VesselSegment(TEST_PATH_LINE, voxel_size)
            nose.tools.assert_equal(vs_line.get_hausdorff_dimension(), 1)

            vs = vessel_stats.VesselSegment(TEST_PATH_CYCLE, voxel_size)
            nose.tools.assert_equal(vs.get_hausdorff_dimension(), -0.0)

        # TODO(meawoppl) this deserves a more sophisticated test.

    def test_vesselsegment_get_vessel_stats(self):
        vs_line = vessel_stats.VesselSegment(TEST_PATH_LINE)
        nose.tools.assert_dict_equal(
            {
                "nodes": 5,
                "length": 4,
                "contraction": 1,
                "start_point": TEST_PATH_LINE[-1],
                "end_point": TEST_PATH_LINE[0],
            },
            vs_line.get_vessel_stats()
        )
