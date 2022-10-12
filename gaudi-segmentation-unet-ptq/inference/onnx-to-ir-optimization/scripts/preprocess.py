'''
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

# Portions of this software are copyright of their respective authors and
# released under the Apache license:
- HabanaAI/Model-References,
  Copyright (c) 2021 Habana Labs, Ltd. an Intel Company.

For licensing see:
https://github.com/HabanaAI/Model-References/blob/master/
PyTorch/computer_vision/segmentation/Unet/LICENSE and
https://github.com/HabanaAI/Model-References/blob/master/
PyTorch/computer_vision/segmentation/Unet/preprocess.py
--------------------------------------------------------------------------
'''

import numpy as np
from monai import transforms
import math

patch_size = [192, 160]


def pad(image, padding):
    """
    pad image
    """
    pad_d, pad_w, pad_h = padding
    return np.pad(
        image,
        (
            (0, 0),
            (math.floor(pad_d), math.ceil(pad_d)),
            (math.floor(pad_w), math.ceil(pad_w)),
            (math.floor(pad_h), math.ceil(pad_h)),
        ),
    )


def calculate_pad_shape(image):
    """
    calculate pad shape based on patch size
    """
    min_shape = patch_size[:]
    image_shape = image.shape[1:]
    if len(min_shape) == 2:  # In 2D case we don't want to pad depth axis.
        min_shape.insert(0, image_shape[0])
    pad_shape = [max(mshape, ishape)
                 for mshape, ishape in zip(min_shape, image_shape)]
    return pad_shape


def standardize_layout(data):
    """
    standardize layout for image/label
    """
    if len(data.shape) == 3:
        data = np.expand_dims(data, 3)
    return np.transpose(data, (3, 2, 1, 0))


def standardize(image, label):
    """
    standardize image shape with paddings
    """
    pad_shape = calculate_pad_shape(image)
    image_shape = image.shape[1:]
    if pad_shape != image_shape:
        paddings = [(pad_sh - image_sh) / 2 for (pad_sh, image_sh)
                    in zip(pad_shape, image_shape)]
        image = pad(image, paddings)
        label = pad(label, paddings)
    _, _, height, weight = image.shape
    start_h = (height - patch_size[0]) // 2
    start_w = (weight - patch_size[1]) // 2
    image = image[:, :, start_h: start_h +
                  patch_size[0], start_w: start_w + patch_size[1]]
    label = label[:, :, start_h: start_h +
                  patch_size[0], start_w: start_w + patch_size[1]]
    return image, label


def apply(image, label, psize=None):
    """
    Apply preprocessing on image and label with optional patch_size
    """
    if psize is not None:
        global patch_size
        patch_size = psize
    crop_fn = transforms.CropForegroundd(
        keys=["image", "label"], source_key="image")
    image = standardize_layout(image)
    label = standardize_layout(label)
    data = crop_fn({"image": image, "label": label})
    image, label = data["image"], data["label"]

    normalize_fn = transforms.NormalizeIntensity(
        nonzero=False, channel_wise=True)
    image = normalize_fn(image)

    image, label = standardize(image, label)
    return image, label
