import cv2
import numpy as np


def get_centroid(contour):
    """
    Returns overlay the 'image' with the skeleton in red
    """
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
