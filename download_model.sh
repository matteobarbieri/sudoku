#!/bin/bash

# Create folder if it does not exist
mkdir -p models

# Download model from dropbox
wget https://www.dropbox.com/s/t0i4x9bgaml3nra/resnet101_allfonts_mnist.pth?dl=1 \
    -O models/resnet101_allfonts_mnist.pth
