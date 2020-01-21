# Cell tracking 
Cell tracking in fluoroscent images after binary segmentation using opencv and calculating statistics per cell after obtaining contours and skeleton of a cell while tracking a cell simultaneously.

To run this code you need to have a binary image directory which is given to the flat `annotation_dir` and the results of tracking where each different cell is colored uniquely per frame and same color is maintained for the cell if it persists throughout different frames are stored in `tracking_dir` and `skeleton_dir` stores one pixel centerlines of all cells in an image. 

python3 cell_tracking/object_tracker.py --annotation_dir=binary_images/ --tracking_dir=tracked_images/ --skeleton_dir=skeletonization_results/ --max_disappeared=5
