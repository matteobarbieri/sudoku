from torch.utils.data import Dataset

import torch

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import glob
import random


class AddSPNoise(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, tensor):
        sp = (torch.rand(tensor.size()) < self.prob) * tensor.max()
        return tensor + sp

    def __repr__(self):
        return self.__class__.__name__ + "(prob={0})".format(self.prob)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class PrintedMNIST(Dataset):
    """Generates images containing a single digit from font"""

    def __init__(self, N, random_state, transform=None):
        """"""
        self.N = N
        self.random_state = random_state
        self.transform = transform

        fonts_folder = "fonts"

        # self.fonts = ["Helvetica-Bold-Font.ttf", 'arial-bold.ttf']
        self.fonts = glob.glob(fonts_folder + "/*.ttf")

        random.seed(random_state)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        target = random.randint(0, 9)

        size = random.randint(150, 250)
        x = random.randint(30, 90)
        y = random.randint(30, 90)

        color = random.randint(200, 255)

        # Generate image
        img = Image.new("L", (256, 256))

        target = random.randint(0, 9)

        size = random.randint(150, 250)
        x = random.randint(30, 90)
        y = random.randint(30, 90)

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(random.choice(self.fonts), size)
        draw.text((x, y), str(target), color, font=font)

        img = img.resize((28, 28), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)

        return img, target
