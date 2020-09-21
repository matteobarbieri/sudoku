from PIL import ImageFont, ImageDraw

import torchvision

import torch.nn as nn


def get_model(model_name, pretrained=True):
    if model_name == "resnet50":

        net = torchvision.models.resnet50(pretrained=pretrained)

        # Replace 1st layer to use it on grayscale images
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    if model_name == "resnet101":

        net = torchvision.models.resnet101(pretrained=pretrained)

        # Replace 1st layer to use it on grayscale images
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    return net


def fill_in_img(resized, original_digits, solved_digits):

    draw = ImageDraw.Draw(resized)

    font = ImageFont.truetype("fonts/Helvetica-Bold-Font.ttf", 40)

    w, h = resized.size
    w9, h9 = w/9, h/9

    offset_x = w9/2 - 15
    offset_y = h9/2 - 15

    # print(original_digits)
    # print(solved_digits)

    for i in range(9):
        for j in range(9):

            if original_digits[i, j] == 0:
                d = str(solved_digits[i, j])

                x = j*w9 + offset_x
                y = i*w9 + offset_y

                draw.text(
                    (x, y), d, color=0, font=font, align="center")

    return resized
