import operator

import cv2
import matplotlib.pyplot as plt
import numpy as np


def pre_process_image(img, skip_dilate=True):
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

    #     plt.imshow(proc, cmap='gray')
    #     plt.title('pre_process_image')
    #     plt.show()
    return proc


def find_corners_of_largest_polygon(img):
    """Finds the 4 extreme corners of the largest contour in the image."""

    # Find contours
    # _, contours, h = cv2.findContours(
    #    img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, h = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours

    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )  # Sort by area, descending
    polygon = contours[0]  # Largest image

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the
    # index of the point.
    # Each point is an array of 1 coordinate, hence the [0] getter, then [0]
    # or [1] used to get x and y respectively.

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
    plt.imshow(warp, cmap="gray")
    plt.title("warp_image")
    plt.show()
    return warp
