import operator

import cv2
import numpy as np
from skimage import measure


def pre_process_image(img: np.ndarray, skip_dilate: bool=True):
    """
    Uses a blurring function, adaptive thresholding and dilation to expose
    the main features of an image.
    """

    # Gaussian blur with a kernal size (height, width) of 9.
    # Note that kernal sizes must be positive and odd and the kernel must be
    # square.
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc)

    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8
        )
        proc = cv2.dilate(proc, kernel)

    return proc


def find_corners_of_largest_polygon(img: np.ndarray):
    """Finds the 4 extreme corners of the largest contour in the image."""

    # Find contours
    contours, h = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area, descending
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )  
    polygon = contours[0]  # Largest image

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )
    top_right, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1),
    )

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [
        polygon[top_left][0],
        polygon[top_right][0],
        polygon[bottom_right][0],
        polygon[bottom_left][0],
    ]


def crop_and_warp(img, crop_rect):
    """
    Crops and warps a rectangular section from an image into a square of
    similar size.
    """

    # Rectangle described by top left, top right, bottom right and bottom left
    # points.
    top_left, top_right, bottom_right, bottom_left = (
        crop_rect[0],
        crop_rect[1],
        crop_rect[2],
        crop_rect[3],
    )

    # Explicitly set the data type to float32 or `getPerspectiveTransform`
    # will throw an error.
    src = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype="float32"
    )

    def distance_between(x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    # Get the longest side in the rectangle
    side = max(
        [
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right),
        ]
    )

    # Describe a square with side of the calculated length, this is the new
    # perspective we want to warp to.
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype="float32",
    )

    # Gets the transformation matrix for skewing the image to fit a square by
    # comparing the 4 before and after points.
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    warp = cv2.warpPerspective(img, m, (int(side), int(side)))
    return warp


def remove_stuff(img: np.ndarray):

    h, w = img.shape

    h10 = int(h/10)
    w10 = int(w/10)

    # Make a copy
    img_cp = np.array(img)

    img_cp = img_cp[h10:, w10:]
    img_cp = img_cp[:-h10, :-w10]

    return img_cp


def remove_stuff_old(img):
    """
    Deprecated
    """
    labels, n_labels = measure.label(img, background=700, return_num=True)

    max_i = -1
    max_i_count = -1

    for i in range(1, n_labels + 1):
        a = (labels == i).sum()

        if a > max_i_count:
            max_i_count = a
            max_i = i

    # Get the shape of the image
    h, w = labels.shape

    # Isolate stuff
    mask = labels == max_i

    i_start = i_end = j_start = j_end = -1

    i = 0
    while i < h:
        if mask[i, :].sum() > 0:
            i_start = i
            break
        else:
            i += 1

    i = h - 1
    while i >= 0:
        if mask[i, :].sum() > 0:
            i_end = i
            break
        else:
            i -= 1

    j = 0
    while j < w:
        if mask[:, j].sum() > 0:
            j_start = j
            break
        else:
            j += 1

    j = w - 1
    while j >= 0:
        if mask[:, j].sum() > 0:
            j_end = j
            break
        else:
            j -= 1

    # For some reason it's best to remove one extra pixel from each side
    i_start += 1
    i_end -= 1

    j_start += 1
    j_end -= 1

    out = img[i_start:i_end, j_start:j_end]

    return out
