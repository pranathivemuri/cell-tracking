from centroid_tracker import CentroidTracker
import argparse
import natsort
import glob
import cv2
import numpy as np


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
    moments = cv2.moments(xy)
    moments = moments
    contour_stats["centroid"] = (
        int(moments['m10'] / moments['m00']),
        int(moments['m01'] / moments['m00']))
    contour_stats["aspect_ratio"] = float(w) / h
    contour_stats["boxed_centroid"] = (centroid[0] - x, centroid[1] - y)
    contour_stats["perimeter"] = cv2.arcLength(xy, True)
    contour_stats["extent"] = float(filled_area) / (w * h)
    convex_hull = cv2.convexHull(xy).squeeze(axis=1)
    contour_stats["solidity"] = filled_area / cv2.contourArea(convex_hull)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Track frames saved as pngs of a video in a given folder. Usage:" +
        "python3 object_tracker.py --annotation_dir=/home/pranathi/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Annotated_Contours_Filled/")
    parser.add_argument(
        "--annotation_dir",
        help="Absolute path to predicted binary annotation images labels should be 0, 255, or 0, 1 etc", required=True, type=str)

    args = parser.parse_args()
    annotation_dir = args.annotation_dir
    annotation_files = natsort.natsorted(glob.glob(annotation_dir + "*.png"))

    # initialize our centroid tracker and binary_image dimensions
    centroid_tracker = CentroidTracker()
    (H, W) = (None, None)
    # if the binary_image dimensions are None, grab them
    binary_image = cv2.imread(annotation_files[0], cv2.IMREAD_UNCHANGED)
    if W is None or H is None:
        (H, W) = binary_image.shape[:2]

    # loop over the images in the prediction folder
    for path in annotation_files:
        binary_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        _, contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        centroids = []

        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        # loop over the detections
        for contour in contours:
            # Drawing rectangle around the contour
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            binary_image = cv2.rectangle(binary_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(contour) != 1:
                moments = cv2.moments(contour)
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                centroids.append(tuple((cx, cy)))
        # update our centroid tracker using the computed set of rectangles
        objects = centroid_tracker.update(centroids)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output binary_image
            text = "ID {}".format(objectID)
            cv2.putText(
                binary_image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.circle(binary_image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # show the output binary_image
        cv2.imshow("binary_image", binary_image)
        # 1000 = delay in milliseconds
        key = cv2.waitKey(1000) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
