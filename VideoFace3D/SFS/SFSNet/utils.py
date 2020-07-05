# coding=utf8
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np


def _convert(src, max_value):
    # find min and max
    _min = np.min(src)
    _max = np.max(src)
    # scale to (0, max_value)
    dst = (src - _min) / (_max - _min + 1e-10)
    dst *= max_value
    return dst


def convert(src, dtype=np.uint8, max_value=255.0):
    # type: (np.ndarray, object, float) -> np.ndarray
    # copy src
    dst = src.copy()
    if src.ndim == 2:
        dst = _convert(dst, max_value)
    elif src.ndim == 3:
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
        light_channel = _convert(dst[0], max_value)
        dst[0, ...] = light_channel
        dst = cv2.cvtColor(dst, cv2.COLOR_LAB2BGR)*255
    else:
        raise RuntimeError("src/utils.py(30): src.ndim should be 2 or 3")
    return dst.astype(dtype)
