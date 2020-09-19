import numpy as np

import torch

# from flask import Flask, flash, request, redirect, url_for
from flask import Flask, request, jsonify

from werkzeug.utils import secure_filename

import base64
from io import BytesIO

import os

from PIL import Image

# FIXME cheap trick to include code from parent folder, fix
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2

# Sudoku-specific imports
from train_classifier import get_model
from data_preprocessing import (
    crop_and_warp, find_corners_of_largest_polygon,
    pre_process_image, remove_stuff)

from classifier import smart_classify

from sudoku_solver import print_board, solve_array

from utils import fill_in_img

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = 'cuda'

net = get_model("resnet101")
model_path = os.path.join(
    "..", "models", "resnet101_allfonts_mnist.pth")

net.load_state_dict(torch.load(model_path))
net = net.to(device)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/sudoku', methods=['POST'])
def sudoku():

    # TODO retrieve image from POST data

    if 'the-file' not in request.files:
        # flash('No file part')
        return "No file!"
        # return redirect(request.url)

    file = request.files['the-file']

    # if user does not select file, browser also
    # submit an empty part without filename
    # if file.filename == '':
        # flash('No selected file')
        # return redirect(request.url)

    if not allowed_file(file.filename):
        return jsonify({'a': 'e'})

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Load image
    img = cv2.imread(image_path)

    # TODO remove here
    # image_path

    # Convert to grayscale
    # TODO maybe check the color format beforehand? [RGB/RGBA/GRAY]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pre_preocessed = pre_process_image(gray)

    # Isolate area of the grid
    corners = find_corners_of_largest_polygon(pre_preocessed)
    cropped = crop_and_warp(pre_preocessed, corners)

    # ## Solve sudoku!

    # Threshold the cropped image
    _, cropped_t = cv2.threshold(cropped, 127, 255, cv2.THRESH_TOZERO)
    cropped = cropped_t

    def slice_grid(grid, i, j, W):
        """
        Assuming grid is an array representing the image, return a single
        square based on its coordinates (which start from 0).
        """

        i_start = i * W
        i_end = i_start + W

        j_start = j * W
        j_end = j_start + W

        aa = grid[i_start:i_end, j_start:j_end]
        return aa

    # Retrieve the shape of the image
    w, h = cropped.shape

    W = w//9

    digits = list()

    for i in range(9):

        for j in range(9):

            # Slice a cell from the original image
            aa = slice_grid(cropped_t, i, j, W)

            bb = aa.astype(float)/255

            # Isolate digit from rest of the cell
            out = remove_stuff(bb)

            # Resize to 28x28
            resized = cv2.resize(out, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(smart_classify(resized, net, conf_threshold=0.9))

    def convert(d):
        if d == ' ':
            return 0
        else:
            return int(d)

    digits_int = np.array([convert(d) for d in digits])
    digits_int = digits_int.reshape((9, 9))

    digits_list = [list(ll) for ll in list(digits_int)]

    # Print recognized digits
    print_board(digits_list)

    # Solve sudoku!
    digits_list = list(digits_list)
    original_digits_arr = np.array(digits_list)
    solve_array(digits_list)

    # Print solved board
    print_board(digits_list)

    # with open(image_path, "rb") as image_file:
        # encoded_string = base64.b64encode(image_file.read())

    buffer_original = BytesIO()
    buffer_solved = BytesIO()

    resized = cv2.resize(
        255-cropped_t, (512, 512), interpolation=cv2.INTER_AREA)
    resized_pil = Image.fromarray(resized)

    # Fill in the rest of the digits
    filled_in_img = fill_in_img(
        resized_pil.copy(), original_digits_arr,
        np.array(digits_list))

    resized_pil.save(buffer_original, format="JPEG")
    original_img_base64 = base64.b64encode(buffer_original.getvalue())

    filled_in_img.save(buffer_solved, format="JPEG")
    solved_img_base64 = base64.b64encode(buffer_solved.getvalue())

    # TODO delete image after using it
    os.remove(image_path)

    output = {
        'img_orig_base64': original_img_base64.decode("utf-8"),
        'img_solved_base64': solved_img_base64.decode("utf-8"),
    }

    return jsonify(output)


@app.route('/hello', methods=['GET'])
def hello():
    return {'hello': 'world'}
