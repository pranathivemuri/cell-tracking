import cv2
import re
import numpy as np
import glob
import natsort
import os
import copy
import scipy.ndimage
import skimage.morphology


"""
This program goes through groundtruth annotations with each instance of cell marked in a different color 1) separtes two touching cells
2) takes out 255 white pixels aroung the contours and sets them to the right color, 3) fills holes in any cells using closing operation
4) fills holes manually when cannot using matplotlib 5) removes small processes 6) replaces incorrect colors with the constant correct colors
7) makes sure expected number of cells == number of colors in ground truth 8) saves the corrected colored images, grayscale instance labeled image and binary images
"""
path = "/Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Contours_Tracked/"
png_files = natsort.natsorted(glob.glob(path + "*.png"))

tracks_corrected_path = "/Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled"
labeled_path = "/Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Cells_Labeled"
binary_path = "/Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Cells_Binary"

for frame, png in enumerate(png_files):

    img = cv2.imread(png)
    img_copy = copy.deepcopy(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_label, num_labels = scipy.ndimage.measurements.label(img_gray, scipy.ndimage.generate_binary_structure(2, 2).astype(np.uint8))
    print(png)
    for label in range(num_labels):
        # find boundaries of the crowded region
        mask = np.zeros_like(img_label)
        mask[img_label == label + 1] = 1
        gray_region = img_gray * mask
        color_region = img * np.dstack([mask] * 3)
        unique_gray_elements = np.unique(gray_region).tolist()
        unique_color_elements = (np.unique(color_region.reshape(-1, color_region.shape[2]), axis=0)).tolist()

        if 0 in unique_gray_elements:
            unique_gray_elements.remove(0)
            unique_color_elements.remove([0, 0, 0])

        if 255 in unique_gray_elements:
            unique_gray_elements.remove(255)
            unique_color_elements.remove([255, 255, 255])

        if len(unique_gray_elements) == 1:
            print("Processing label under normal non-touching cells {}".format(label))
            # Fill with non-white color in img
            if len(unique_color_elements) != 1:
                img_label_region = img_label * mask
                index = list(set(map(tuple, np.transpose(np.where(img_label_region == label + 1)))))[0]
                unique_color_element = color_region[index[0], index[1], :]
            else:
                unique_color_element = unique_color_elements[0]
            for i in range(3):
                color_region[:, :, i][color_region[:, :, i] != 0] = unique_color_element[i]
                color_region[:, :, i] = scipy.ndimage.grey_closing(color_region[:, :, i], size=(3, 3))
            nonzeros = list(set(map(tuple, np.transpose(np.nonzero(color_region)))))
            for index in nonzeros:
                for i in range(3):
                    img_copy[index[0], index[1], i] = color_region[index[0], index[1], i]
        elif len(unique_gray_elements) == 0:
            nonzeros = list(set(map(tuple, np.transpose(np.nonzero(color_region)))))
            for index in nonzeros:
                for i in range(3):
                    img_copy[index[0], index[1], i] = 0

        else:
            weird_touching_cells = False
            print("bad cluster in the middle")
            print("Processing label under abnormal touching cells {}".format(label))
            # remove 255s to separate the cells
            bad_color_region = copy.deepcopy(color_region)
            bad_gray_region = copy.deepcopy(gray_region)
            num_cells = len(unique_color_elements)
            print("number of colors in one sub label {}".format(num_cells))
            mask = np.zeros_like(bad_gray_region)
            mask[bad_gray_region == 255] = 1
            for i in range(3):
                bad_color_region[:, :, i][mask != 0] = 0
                img_copy[:, :, i][mask != 0] = 0
            img_gray[mask != 0] = 0
            bad_gray_region[mask != 0] = 0
            img_sub_label, num_sub_labels = scipy.ndimage.measurements.label(bad_gray_region, scipy.ndimage.generate_binary_structure(2, 2).astype(np.uint8))

            print("number of labels in one sub label {}".format(num_sub_labels))

            if num_cells == 2 and num_sub_labels == 1:
                weird_touching_cells = True
                img_sub_label, num_sub_labels = scipy.ndimage.measurements.label(bad_gray_region, scipy.ndimage.generate_binary_structure(2, 1).astype(np.uint8))

            gray_elements_dict = {}
            color_elements_dict = {}
            bad_color_region_copy = copy.deepcopy(bad_color_region)

            for sub_label in range(num_sub_labels):
                print("Processing sub_label {}".format(sub_label))
                sub_mask = np.zeros_like(img_label)
                sub_mask[img_sub_label == sub_label + 1] = 1
                sub_gray_region = img_gray * sub_mask
                sub_color_region = img_copy * np.dstack([sub_mask] * 3)
                img_sub_label_region = img_sub_label * sub_mask
                index = list(set(map(tuple, np.transpose(np.where(img_sub_label_region == sub_label + 1)))))[0]
                unique_gray_element = sub_gray_region[index]
                unique_color_element = sub_color_region[index[0], index[1], :]
                gray_elements_dict[sub_label + 1] = unique_gray_element
                color_elements_dict[sub_label + 1] = unique_color_element

            print(gray_elements_dict)
            print(color_elements_dict)

            elements = list(gray_elements_dict.values())
            for sub_label, grayscale_value in gray_elements_dict.items():
                if elements.count(grayscale_value) == 1:
                    # when the colors didn't screw up
                    sub_mask = np.zeros_like(img_label)
                    sub_mask[img_sub_label == sub_label] = 1
                    sub_gray_region = img_gray * sub_mask
                    closed = scipy.ndimage.grey_closing(sub_gray_region, size=(3, 3))

                    closed_color = np.zeros_like(bad_color_region)
                    closed_color[closed != 0] = color_elements_dict[sub_label]

                    # connect two components of same color
                    nonzeros = list(set(map(tuple, np.transpose(np.nonzero(closed_color)))))
                    for i in range(3):
                        for index in nonzeros:
                            img_copy[index[0], index[1], i] = closed_color[index[0], index[1], i]
                    bad_gray_region[bad_gray_region == grayscale_value] = 0

            for sub_label, grayscale_value in gray_elements_dict.items():
                if elements.count(grayscale_value) == 1 and weird_touching_cells:
                    print("processed as weird_touching_cells")
                    # when the colors didn't screw up
                    sub_mask = np.zeros_like(img_label)
                    sub_mask[img_sub_label == sub_label] = 1
                    sub_gray_region = img_gray * sub_mask
                    thinned = skimage.morphology.thin(sub_gray_region, 2)

                    closed_color = np.zeros_like(bad_color_region)
                    closed_color[thinned != 0] = color_elements_dict[sub_label]

                    # connect two components of same color
                    nonzeros = list(set(map(tuple, np.transpose(np.nonzero(closed_color)))))
                    for i in range(3):
                        for index in nonzeros:
                            img_copy[index[0], index[1], i] = closed_color[index[0], index[1], i]
                    bad_gray_region[bad_gray_region == grayscale_value] = 0

            for sub_label, grayscale_value in gray_elements_dict.items():
                if elements.count(grayscale_value) == 2:
                    screwed_up_color = color_elements_dict[sub_label]

                    dilated_bad_gray_region = skimage.morphology.dilation(bad_gray_region)
                    thinned = skimage.morphology.thin(dilated_bad_gray_region, 1)
                    thinned = scipy.ndimage.grey_closing(thinned, size=(3, 3))

                    thinned_color = np.zeros_like(bad_color_region)
                    thinned_color[thinned != 0] = screwed_up_color

                    # connect two components of same color
                    nonzeros = list(set(map(tuple, np.transpose(np.nonzero(thinned_color)))))
                    for i in range(3):
                        for index in nonzeros:
                            img_copy[index[0], index[1], i] = thinned_color[index[0], index[1], i]
    img_label, num_labels = scipy.ndimage.measurements.label(cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY), scipy.ndimage.generate_binary_structure(2, 2).astype(np.uint8))
    unique_color_elements = (np.unique(img_copy.reshape(-1, img_copy.shape[2]), axis=0)).tolist()
    if num_labels != len(unique_color_elements) - 1:
        print("BAD PNG {}".format(png))
        print("Needs removing small processes")

        for label in range(num_labels):
            print("Post processing label {}".format(label))
            mask = np.zeros_like(img_label)
            mask[img_label == label + 1] = 1

            print(mask.sum())

            if mask.sum() < 20:
                for i in range(3):
                    img_copy[:, :, i][mask != 0] = 0
    img_label, num_labels = scipy.ndimage.measurements.label(cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY), scipy.ndimage.generate_binary_structure(2, 2).astype(np.uint8))
    if num_labels != len(unique_color_elements) - 1:
        print("BADDEST PNG {}".format(png))

    save_path = os.path.join(tracks_corrected_path, os.path.basename(png))
    save_path = save_path.replace(".png", "_corrected.png")
    cv2.imwrite(save_path, img_copy)


# Special cases where added for these conditions in above program after equating number of cells
# 9 - disconnected small process remove
# 14 - same pink color instead of blue and pink correct
# 19 - disconnected small process remove
# 22 - same blue color instead of blue and pink correct
# 24 - disconnected small process remove
# 25 - disconnected small process remove
# 32 - disconnected small process remove
# 57 - disconnected small process remove
# 59 - disconnected small process remove


# The following frames were edited in python by manually setting pixels to zeros and needed colors using matplotlib
# 14 - wrong number of objects pink and yellow touching still considered one, remove single pixel connecting them
# 22 - wrong number of objects pink and yellow touching still considered one, remove single pixel connecting them
# 24 - couldn't remove the small process, connect them instead
# 25 - couldn't remove the small process, connect them instead

png_files = natsort.natsorted(glob.glob(tracks_corrected_path + "/*.png"))
for png in png_files:
    img = cv2.imread(png)
    unique_color_elements = (np.unique(img.reshape(-1, img.shape[2]), axis=0)).tolist()
    img_label, num_labels = scipy.ndimage.measurements.label(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), scipy.ndimage.generate_binary_structure(2, 2).astype(np.uint8))
    print(png, unique_color_elements)
    assert len(unique_color_elements) - 1 == num_labels

# correct colors for cells
# /Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled/SUM_20180504_Micr_GW23_GFP_TimeLapse_001.Pos001-14-tracking_corrected.png [[0, 0, 0], [0, 255, 0], [0, 255, 255], [202, 191, 254], [218, 111, 146], [255, 0, 0], [255, 0, 255], [255, 255, 83]]
# /Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled/SUM_20180504_Micr_GW23_GFP_TimeLapse_001.Pos001-27-tracking_corrected.png [[0, 0, 0], [0, 0, 255], [0, 255, 0], [83, 255, 255], [202, 191, 254], [218, 111, 146], [255, 0, 0], [255, 255, 0]]
# /Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled/SUM_20180504_Micr_GW23_GFP_TimeLapse_001.Pos001-36-tracking_corrected.png [[0, 0, 0], [0, 0, 255], [0, 255, 0], [83, 255, 255], [202, 191, 254], [218, 111, 146], [255, 0, 0], [255, 255, 0]]
# /Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled/SUM_20180504_Micr_GW23_GFP_TimeLapse_001.Pos001-55-tracking_corrected.png [[0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [202, 191, 254], [218, 111, 146], [255, 0, 0], [255, 0, 255], [255, 255, 83]]
# /Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled/SUM_20180504_Micr_GW23_GFP_TimeLapse_001.Pos001-89-tracking_corrected.png [[0, 0, 0], [0, 0, 255], [0, 255, 0], [83, 255, 255], [152, 141, 145], [202, 191, 254], [218, 111, 146], [255, 0, 0], [255, 255, 0]]
# /Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled/SUM_20180504_Micr_GW23_GFP_TimeLapse_001.Pos001-108-tracking_corrected.png [[0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [202, 191, 254], [218, 111, 146], [255, 0, 0], [255, 0, 255], [255, 255, 83]]
# /Users/pranathivemuri/Documents/Volumes/MicroscopyData/imaging_group/ucsf_microglia/UCSF-2018-05-04-00-00-00-0001_Corrected_Labeled/SUM_20180504_Micr_GW23_GFP_TimeLapse_001.Pos001-119-tracking_corrected.png [[0, 0, 0], [0, 0, 255], [0, 255, 0], [83, 255, 255], [151, 141, 144], [202, 191, 254], [218, 111, 146], [255, 0, 0], [255, 0, 255], [255, 255, 0]]

# 152, 141, 145 color wrong and color value (83) incorrect in a channel
# 203, 192, 255
# 219, 112, 147 wrong colors as well more by one pixel
# bad_colors = [(203, 192, 255), (219, 112, 147)]
# good_colors = [(202, 191, 254), (218, 111, 146)]
# 0, 64, 255 instead of 0, 0, 255

# for png in png_files[0:6]:
#     img = cv2.imread(png)
#     for index, color in enumerate(bad_colors):
#         color_array = np.array(color)
#         mask = cv2.inRange(img, color_array, color_array)

#         for i in range(3):
#             img[:, :, i][mask != 0] = good_colors[index][i]

#     cv2.imwrite(png, img)

# After changing colors testing again

png_files = natsort.natsorted(glob.glob(tracks_corrected_path + "/*.png"))
for png in png_files:
    img = cv2.imread(png)
    unique_color_elements = (np.unique(img.reshape(-1, img.shape[2]), axis=0)).tolist()
    img_label, num_labels = scipy.ndimage.measurements.label(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), scipy.ndimage.generate_binary_structure(2, 2).astype(np.uint8))
    print(png, unique_color_elements)
    assert len(unique_color_elements) - 1 == num_labels

# naming cells 0, 1, --
# From Galina Ground truth palette on GIMP below translated to same colors but different intensities when saved
# 0 blue: 0, 0, 235; 1 green: 0, 205, 0; 2 red: 255, 0, 0; 3 teal: 0, 255, 255; 4 magenta: 255, 0, 255; 5 yellow: 255, 255, 0; 6 pink: 100, 75.3, 79.6; 7 violet: 57.6, 43.9, 85.9; 8 grey: 56.7, 55.4, 59.4


BLUE = [0, 0, 255]
GREEN = [0, 255, 0]
RED = [255, 0, 0]
TEAL = [0, 255, 255]
MAGENTA = [255, 255, 0]
YELLOW = [255, 0, 255]
PINK = [218, 111, 146]
PURPLE = [202, 191, 254]
GREY = [151, 141, 144]


BLUE_LABEL = "cell0"
GREEN_LABEL = "cell1"
RED_LABEL = "cell2"
TEAL_LABEL = "cell3"
MAGENTA_LABEL = "cell4"
YELLOW_LABEL = "cell5"
PINK_LABEL = "cell6"
PURPLE_LABEL = "cell7"
GREY_LABEL = "cell8"

CELL_ANNOTATIONS_COLORS_LABELS_DICT = dict(cell0=BLUE,
                                           cell1=GREEN,
                                           cell2=RED,
                                           cell3=TEAL,
                                           cell4=MAGENTA,
                                           cell5=YELLOW,
                                           cell6=PINK,
                                           cell7=PURPLE,
                                           cell8=GREY)


for png in png_files:
    img = cv2.imread(png)
    # grayscale image contains integers 1, 2, 3,
    grayscale_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    unique_color_elements = (np.unique(img.reshape(-1, img.shape[2]), axis=0)).tolist()
    unique_color_elements.remove([0, 0, 0])

    # set each color to the cell number as in the above dictionary, get the grayscale image
    for color in unique_color_elements:
        color_array = np.array(color)
        mask = cv2.inRange(img, color_array, color_array)
        output = cv2.bitwise_and(img, img, mask)
        cell = [label for label, const_color in CELL_ANNOTATIONS_COLORS_LABELS_DICT.items() if color == const_color][0]
        grayscale_label = int(re.search(r'\d+', cell).group())
        grayscale_image[mask != 0] = grayscale_label + 1

    # binary image contains 0 for backround 255 for cells
    binary_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    binary_image[grayscale_image != 0] = 255

    #  equating number of unique gray elements - 1 = number of binary objects
    unique_gray_elements = (np.unique(grayscale_image)).tolist()
    unique_gray_elements.remove(0)

    img_label, num_labels = scipy.ndimage.measurements.label(binary_image, scipy.ndimage.generate_binary_structure(2, 2).astype(np.uint8))
    assert len(unique_color_elements) == len(unique_gray_elements) == num_labels, "png {}".format(png)

    # save grayscale images
    save_path = os.path.join(labeled_path, os.path.basename(png))
    save_path = save_path.replace(".png", "_grayscale.png")
    cv2.imwrite(save_path, grayscale_image)

    # save grayscale images
    save_path = os.path.join(binary_path, os.path.basename(png))
    save_path = save_path.replace(".png", "_binary.png")
    cv2.imwrite(save_path, binary_image)

