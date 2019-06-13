import sys
sys.path.append(".")  # NOQA


import argparse
import natsort
import time
import glob
import cv2
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import skimage.morphology

from cell_tracking.centroid_tracker import CentroidTracker
import cell_tracking.skeleton_graph_stats as skeleton_stats
import cell_tracking.contour_stats as contour_features


SKELETON_COLOR = [255, 0, 0]
OVERLAY_ALPHA = 0.7
CELL_COLOR = [0, 255, 0]
blue = [0, 0, 235]
green = [0, 205, 0]
red = [255, 0, 0]
cyan = [0, 255, 255]
magenta = [255, 0, 255]
yellow = [255, 255, 0]
pink = [255, 192, 203]
purple = [147, 112, 219]

CELL_COLORS = {0: blue, 1: green, 2: red, 3: cyan, 4: magenta, 5: yellow, 6: pink, 7: purple, 8: yellow}


def _get_objects(arr):
    label_skel, count_objects = scipy.ndimage.measurements.label(
        arr, scipy.ndimage.generate_binary_structure(arr.ndim, 2))
    return count_objects


def assert_valid_binary_image(binary_mask):
    """
    Assert to see if the binary_mask is:
    - binary (have more than two values 0,1  or 0, 255) (segmentation masks are always binary)
    """
    unique_values = np.unique(binary_mask).tolist()
    is_binary = (unique_values == [0, 1] or
                 unique_values == [0, 255] or
                 unique_values == [0] or
                 unique_values == [1] or
                 unique_values == [255])
    assert is_binary, \
        "Expected [0, 1] or [0, 255], found incorrect values: {}".format(unique_values)


def get_aggregate_stats(stats):
    """
    Returns aggregate stss i.e reduces the stats per each branch in the skeleton to a list
    so there is stats per whole skeleton in form of dictionary.
    if it has 5 branches has 5 lists for each fo the features
    nodes, length, contraction, start_point, end_point
    Ex: {'length': [1, 2], 'contraction': [1, 1.2], 'start_point':[(1, 2), (3, 4)], 'end_point': [(5, 6), (7, 8)]}
    """
    expected_keys = 'nodes', 'length', 'contraction', 'start_point', 'end_point'
    reduced_segments = {key: [] for key in expected_keys}
    for segment in stats:
        if len(segment) == 1:
            for gloabl_attrs, _ in segment.items():
                reduced_segments[gloabl_attrs] = segment[gloabl_attrs]
            continue
        for key in expected_keys:
            reduced_segments[key] += [segment[key]]
    return reduced_segments


def get_overlapped_skeleton_image(binary_image, skeleton_image):
    """
    Returns overlay the 'binary_image' where with the skeleton is in SKELETON_COLOR and binary_image is in CELL_COLOR
    """
    overlaid_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    overlaid_image[binary_image != 0] = CELL_COLOR

    overlaid_image[skeleton_image != 0] = SKELETON_COLOR
    return cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)


def get_skeleton_stats(binary_image):
    """
    Returns stats per branch after skeletonizing the 'binary_image'
    """
    skeleton_image = skimage.morphology.skeletonize(binary_image)
    stats_object = skeleton_stats.SkeletonStats(skeleton_image, cutoff=0)
    stats, object_lines = stats_object.get_stats_general(stats_object.networkx_graph)
    return get_aggregate_stats(stats)


def dist_between_centroids(centroid1, centroid2):
    """
    Returns euclidean distance between two centroids or cartesean coordinates
    """
    vect = np.array(centroid1, dtype=np.float64) - np.array(centroid2, dtype=np.float64)
    return np.linalg.norm(vect)


def update_trajectory_path_graph(current_cell_ids, trajectory_path_graph, frame):
    """
    Returns updated trajectory path graph. This function loops through the current_cell_ids which contain
    the cell id and centroid. If it doesn't exist, it will initialize them with zeros.
    For cells appearing in not exactly the first frame -
    updates them as with the 'frame' specified as beginning frame
    For other cells updates the trajectory path distances displacement -
    based on the previous trajectory path graph dictionary
    """
    previous_cell_ids = list(trajectory_path_graph.keys())

    # if new cells are formed add more empty rows for tracking
    for current_cell_id, centroid in current_cell_ids.items():
        # initialize trajectory path graph keys only for new cell ids
        if current_cell_id not in previous_cell_ids:
            trajectory_path_graph[current_cell_id] = dict(
                beginning_frame=0,
                ending_frame=0,
                parent_cell=0,
                distance=0,
                starting_centroid=centroid,
                ending_centroid=centroid,
                displacement=0)

    for current_cell_id, centroid in current_cell_ids.items():
        # cell already initialized from a previous frame
        if current_cell_id in previous_cell_ids:
            trajectory_path_graph[current_cell_id]["ending_frame"] = \
                trajectory_path_graph[current_cell_id]["ending_frame"] + 1

        # new cell or a division from parent cell
        else:
            trajectory_path_graph[current_cell_id]["beginning_frame"] = frame
            trajectory_path_graph[current_cell_id]["ending_frame"] = frame + 1

        trajectory_path_graph[current_cell_id]["parent_cell"] = 0
        previous_centroid = trajectory_path_graph[current_cell_id]["ending_centroid"]
        trajectory_path_graph[current_cell_id]["ending_centroid"] = centroid
        trajectory_path_graph[current_cell_id]["distance"] = \
            trajectory_path_graph[current_cell_id]["distance"] + dist_between_centroids(centroid, previous_centroid)
        trajectory_path_graph[current_cell_id]["displacement"] = \
            dist_between_centroids(centroid, trajectory_path_graph[current_cell_id]["starting_centroid"])

    return trajectory_path_graph


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Track frames saved as pngs of a video in a given folder" +
        "and save the pngs with ids overlapped and stats per image in a csv file " +
        "python3 cell_tracking/object_tracker.py" +
        "--annotation_dir=/home/pranathi/UCSF-2018-05-04-00-00-00-0001_Annotated_Contours_Filled/" +
        "--tracking_dir=/home/pranathi//UCSF-2018-05-04-00-00-00-0001_Annotated_Contours_Tracked/" +
        "--skeleton_dir=/home/pranathi//UCSF-2018-05-04-00-00-00-0001_Annotated_Contours_Skeleton/" +
        "--max_disappeared=5" +
        "To generate a video/gif from images in a folder/directory using ffmpeg run" +
        " ffmpeg -framerate 5 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
    parser.add_argument(
        "--annotation_dir",
        help="Absolute path to predicted binary annotation images labels should be 0, 255, or 0, 1 etc",
        required=True, type=str)
    parser.add_argument(
        "--tracking_dir",
        help="Absolute path to tracking overlays",
        required=True, type=str)
    parser.add_argument(
        "--skeleton_dir",
        help="Absolute path to skeleton overlays",
        required=True, type=str)
    parser.add_argument(
        "--max_disappeared",
        help="store the number of maximum consecutive frames a given object" +
        "is allowed to be marked as disappeared until we need to deregister the object from tracking",
        required=True, type=int)

    time_start = time.time()
    args = parser.parse_args()
    tracking_dir = args.tracking_dir
    skeleton_dir = args.skeleton_dir
    annotation_dir = args.annotation_dir

    # list of images in annotation directory
    annotation_files = natsort.natsorted(glob.glob(annotation_dir + "*.png"))

    # initialize our centroid tracker and binary_image dimensions
    centroid_tracker = CentroidTracker(args.max_disappeared)
    binary_image = cv2.imread(annotation_files[0], cv2.IMREAD_GRAYSCALE)
    assert_valid_binary_image(binary_image)
    shape = binary_image.shape[:2]
    (height, width) = shape[:2]

    # initialize trajectory path graph
    trajectory_path_graph = {}

    # loop over the images in the binary annotations folder
    for frame_count, path in enumerate(annotation_files):
        binary_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert_valid_binary_image(binary_image)
        label_im, _ = scipy.ndimage.measurements.label(binary_image, scipy.ndimage.generate_binary_structure(2, 2))
        _, contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        stats_df = []

        # Convert to rgb image once to draw rectangle and write cell ids on the binary image
        rgb_converted_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

        # loop over each cell's contour one at a time to find its centroid, stats.
        for contour_count, contour in enumerate(contours):

            # Contour stats
            contour_stats = contour_features.get_contour_stats(contour)
            centroids.append(contour_stats["centroid"])

            # Skeleton stats
            contour_filled = np.zeros((height, width), dtype=np.uint8)
            contour_filled = cv2.drawContours(contour_filled, [contour], contourIdx=0, color=1, thickness=-1)

            # Drawing rectangle in green around the contour, and filling the contour with object id
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            rectangle_overlaid_image = cv2.rectangle(rgb_converted_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rectangle_overlaid_image[contour_filled != 0] = CELL_COLORS[contour_count]

            # assert number of contours is 1, since we are looking one contour at a time
            assert _get_objects(contour_filled) == 1
            skeletonized_arr_stats = get_skeleton_stats(contour_filled)
            contour_stats.update(skeletonized_arr_stats)
            stats_df.append(contour_stats)

        # update our centroid tracker using the computed set of centroids
        current_cell_ids = centroid_tracker.update(centroids)
        trajectory_path_graph = update_trajectory_path_graph(current_cell_ids, trajectory_path_graph, frame_count)

        # loop over the tracked current_cell_ids
        for (current_cell_id, centroid) in current_cell_ids.items():
            text = "Cell {}".format(current_cell_id)
            cv2.putText(
                rectangle_overlaid_image,
                text,
                (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2)

        # save the image with cells tracked and tagged with their IDs and bounding boxes
        save_path = os.path.join(tracking_dir, os.path.basename(path))
        cv2.imwrite(save_path, cv2.cvtColor(rectangle_overlaid_image, cv2.COLOR_BGR2RGB))

        # save the skeleton image, skeleton is overlaid onto the original binary image
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert_valid_binary_image(image)
        skeleton_image = skimage.morphology.skeletonize(image // 255)
        overlaid_image = get_overlapped_skeleton_image(image, skeleton_image)
        save_path = os.path.join(skeleton_dir, os.path.basename(path))
        cv2.imwrite(save_path, overlaid_image)

        # Save the stats dataframe
        df = pd.DataFrame(stats_df)
        df.to_csv(skeleton_dir + os.path.basename(path).replace(".png", "_") + "contour_skeleton_stats.csv")

    # Save the tracking path dataframe
    trajectory_df = pd.DataFrame(pd.DataFrame(trajectory_path_graph))
    trajectory_df.to_csv(skeleton_dir + "trajectory_path_graph.csv")

    print("Finished tracking %i frames in %0.2f s" % (len(annotation_files), time.time() - time_start))
