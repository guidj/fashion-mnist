"""
Image transformers
"""

from typing import Tuple, Callable

import cv2
import numpy as np

from fmnist import xtype


def create_resize_image_fn(size: Tuple[int, int], new_size: Tuple[int, int],
                           flatten: bool) -> Callable[[xtype.DataTuple], xtype.DataTuple]:
    def resize_image(dt: xtype.DataTuple) -> xtype.DataTuple:
        x, y = dt
        x_resized = cv2.resize(np.reshape(x.astype('uint8'), newshape=size), new_size)
        if flatten:
            return np.reshape(x_resized, newshape=(-1,)), y
        else:
            return x_resized, y

    return resize_image


def normalize_image_values(dt: xtype.DataTuple) -> xtype.DataTuple:
    x, y = dt
    return x.astype('float32') / 255., y


def expand(dt: xtype.DataTuple) -> xtype.DataTuple:
    x, y = dt
    return np.array([x]), y
