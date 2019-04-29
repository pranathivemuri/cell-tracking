from centroid_tracker import CentroidTracker
import argparse
import natsort
import glob
import cv2

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
