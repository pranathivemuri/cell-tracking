from cell_tracking.centroid_tracker import CentroidTracker
import cell_tracking.skeleton_graph_stats as skeleton_stats

import argparse
import natsort
import glob
import cv2
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import skimage.morphology


def _get_objects(arr):
    label_skel, count_objects = scipy.ndimage.measurements.label(
        arr, scipy.ndimage.generate_binary_structure(arr.ndim, 2))
    return count_objects


def get_centroid(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    elif len(contour) == 2:
        cx = (contour[0][0][0] + contour[1][0][0]) // 2
        cy = (contour[0][0][1] + contour[1][0][1]) // 2
    else:
        cx, cy = contour[0][0][0], contour[0][0][1]
    return tuple((cx, cy))


def get_contour_stats(contour):
    """
    Add contour metrics as object attributes for given contour xy.
    This is the primary wrapper for openCV contour attributes

    Raises AssertionError if the last element of contour is not equal to the first
    """
    xy = contour

    x, y, w, h = cv2.boundingRect(contour)

    contour_stats = {}
    filled_area = cv2.contourArea(xy)
    contour_stats["filled_area"] = filled_area
    centroid = get_centroid(xy)
    contour_stats["centroid"] = centroid
    contour_stats["aspect_ratio"] = float(w) / h
    contour_stats["boxed_centroid"] = (centroid[0] - x, centroid[1] - y)
    contour_stats["perimeter"] = cv2.arcLength(xy, True)
    contour_stats["extent"] = float(filled_area) / (w * h)
    convex_hull = cv2.convexHull(xy).squeeze(axis=1)
    if cv2.contourArea(convex_hull) != 0:
        contour_stats["solidity"] = filled_area / cv2.contourArea(convex_hull)
    else:
        contour_stats["solidity"] = 0

    # remove the last coordinate we added to close the loop
    contour_stats["is_convex"] = cv2.isContourConvex(xy[:-1])
    contour_stats["equiv_diameter"] = np.sqrt(4 * filled_area / np.pi)

    # Need at least 5 contour points for an ellipse
    if contour.shape[0] >= 5:
        ellipse = list(cv2.fitEllipse(contour))  # wrap so we return a (mutable) list
        # rotate the angle clockwise by 90 deg to account for tranpose
        ellipse[2] = np.mod(ellipse[2] + 90, 180)

        # if the ellipse is a circle, orientation should be 0
        if ellipse[1][0] == ellipse[1][1]:
            ellipse[2] = 0
    else:
        ellipse = [(0, 0), (0, 0), 0]

    contour_stats["minor_axis_diameter"] = ellipse[1][0]
    contour_stats["major_axis_diameter"] = ellipse[1][1]

    # angle between x-axis and the major axis of the matching ellipse
    contour_stats["orientation"] = ellipse[2]
    return contour_stats


def get_aggregate_stats(stats):
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


def get_skeleton_stats(image):
    skeleton_image = skimage.morphology.skeletonize(image)
    stats_object = skeleton_stats.SkeletonStats(skeleton_image, cutoff=0)
    stats, object_lines = stats_object.get_stats_general(stats_object.networkx_graph)
    return get_aggregate_stats(stats)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Track frames saved as pngs of a video in a given folder" +
        "and save the pngs with ids overlapped and stats per image in a csv file " +
        "python3 cell_tracking/object_tracker.py" +
        "--annotation_dir=/home/pranathi/UCSF-2018-05-04-00-00-00-0001_Annotated_Contours_Filled/" +
        "--results_dir=/home/pranathi//UCSF-2018-05-04-00-00-00-0001_Annotated_Contours_Tracked/" +
        "To generate a video/gif from images in a folder/directory using ffmpeg run" +
        " ffmpeg -framerate 5 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
    parser.add_argument(
        "--annotation_dir",
        help="Absolute path to predicted binary annotation images labels should be 0, 255, or 0, 1 etc",
        required=True, type=str)
    parser.add_argument(
        "--results_dir",
        help="Absolute path to confusion matrix overlays",
        required=True, type=str)

    args = parser.parse_args()
    results_dir = args.results_dir
    annotation_dir = args.annotation_dir
    annotation_files = natsort.natsorted(glob.glob(annotation_dir + "*.png"))

    # initialize our centroid tracker and binary_image dimensions
    centroid_tracker = CentroidTracker()
    binary_image = cv2.imread(annotation_files[0], cv2.IMREAD_UNCHANGED)
    shape = binary_image.shape[:2]
    (height, width) = shape[:2]

    # loop over the images in the prediction folder
    for path in annotation_files:
        binary_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        _, contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        centroids = []

        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        # loop over the detections
        stats_df = []

        for contour in contours:
            # Drawing rectangle around the contour
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            binary_image = cv2.rectangle(binary_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Contour stats
            contour_stats = get_contour_stats(contour)
            centroids.append(contour_stats["centroid"])

            # Skeleton stats
            contour_filled = np.zeros((height, width), dtype=np.uint8)
            contour_filled = cv2.drawContours(contour_filled, [contour], contourIdx=0, color=1, thickness=-1)

            # assert number of objects is 1
            assert _get_objects(contour_filled) == 1
            skeletonized_arr_stats = get_skeleton_stats(contour_filled)
            contour_stats.update(skeletonized_arr_stats)
            stats_df.append(contour_stats)

        # update our centroid tracker using the computed set of rectangles
        objects = centroid_tracker.update(centroids)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output binary_image
            text = "ID {}".format(objectID)
            cv2.putText(
                binary_image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show the output binary_image
        cv2.imshow("binary_image", binary_image)
        save_path = os.path.join(results_dir, os.path.basename(path))

        cv2.imwrite(save_path, binary_image)
        # Save the stats dataframe
        df = pd.DataFrame(stats_df)
        df.to_csv(results_dir + os.path.basename(path).replace(".png", "_") + "contour_skeleton_stats.csv")

        # 1000 = delay in milliseconds
        key = cv2.waitKey(1000) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
