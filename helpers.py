"""
Module contains useful functions =)
"""

import os

from matplotlib.image import imread
from matplotlib.pyplot import figure
from matplotlib.pyplot import imshow
from matplotlib.pyplot import show
from matplotlib.pyplot import subplot
from matplotlib.pyplot import tight_layout
from matplotlib.pyplot import title
from matplotlib.pyplot import xticks
from matplotlib.pyplot import yticks


__ALL__ = ['read_images', 'show_images']


def read_images(path):
    """
    Read all images from path

    Returns a pairs (image_file_name, image)
    """
    for image_file_name in sorted(os.listdir(path)):
        full_image_path = os.path.join(path, image_file_name)
        image = imread(full_image_path)
        yield image_file_name, image


def show_images(images, c=2, figsize=(11, 12)):
    """show_images shows a list of images

    Arguments:
      - images - a list of pairs (image_name, image)
      - c - a number of columns
      - figsize - a size of figure
    """
    r = len(images) // c

    figure(figsize=figsize)
    for n, pair in enumerate(images):
        image_name, image = pair
        subplot(r, c, n + 1)
        imshow(image)
        title(image_name)
        xticks(())
        yticks(())
    tight_layout(pad=0, h_pad=0, w_pad=0)
    show()
