'''
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

# Portions of this software are copyright of their respective authors and
# released under the Apache license:
- IntelAI/unet, Copyright (c) 2019 Intel Corporation.
# For licensing see https://github.com/IntelAI/unet/blob/master/LICENSE
--------------------------------------------------------------------------
'''

from dataloader import DatasetGenerator, get_decathlon_filelist
import numpy as np
from monai.networks import one_hot
import torch
from openvino.runtime import Core
import os
import matplotlib.pyplot as plt
import time
import argparse


def calc_dice(target, prediction, smooth=0.0001):
    """
    Sorensen Dice coefficient
    """
    prediction = np.round(prediction)
    return calc_soft_dice(target, prediction, smooth)


def calc_soft_dice(target, prediction, smooth=0.0001):
    """
    Sorensen (Soft) Dice coefficient - Don't round predictions
    """
    n_pred_ch = prediction.shape[1]

    if target.shape != prediction.shape:
        # Apply OneHot using num_classes on target
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target)
        target = one_hot(target, num_classes=n_pred_ch)
        target = target.numpy()

    # Applying softmax
    prediction = torch.from_numpy(prediction)
    prediction = torch.nn.functional.softmax(prediction, 1)
    prediction = prediction.numpy()

    # Calculate Dice Coefficient
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


data_path = "/home/Optimization/data_source/Task01_BrainTumour"
patch_size = [192, 160]
batch_size = 64
seed = 816
train_test_split = 0.85


trainFiles, validateFiles, testFiles = get_decathlon_filelist(
    data_path=data_path, seed=seed, split=train_test_split)

ds_test = DatasetGenerator(testFiles,
                           batch_size=batch_size,
                           patch_size=patch_size,
                           augment=False,
                           seed=seed)

parser = argparse.ArgumentParser(description='openvino_model_name')
parser.add_argument('--ov_model_name', dest='ov_model_name',
                    type=str, help='openvino_model_name',
                    metavar='ov_model_name', default='')

args = parser.parse_args()
openvino_model_dir = "/home/Optimization/ir_model/"
openvino_model_name = args.ov_model_name

# FP32/INT8 Model
path_to_xml_file = "{}.xml".format(
    os.path.join(openvino_model_dir, openvino_model_name))
path_to_bin_file = "{}.bin".format(
    os.path.join(openvino_model_dir, openvino_model_name))
print(f"OpenVINO IR: {path_to_xml_file}, {path_to_bin_file}")

# Load the network using OpenVINO's 2.0 API's (Latest)
ie = Core()
net = ie.read_model(model=path_to_xml_file, weights=path_to_bin_file)

ov_config = {"PERFORMANCE_HINT": "THROUGHPUT"}
exec_net = ie.compile_model(model=net, device_name="CPU", config=ov_config)
output_layer = exec_net.output(0)


def plot_results(ds):

    img, msk = next(ds.ds)
    # print("img.shape: ", img.shape) #(n,c,h,w) = (64,4,192,160)
    # find the slice with the largest tumor
    idx = np.argmax(np.sum(np.sum(msk[:, 0, :, :], axis=1), axis=1))

    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(img[idx, 0, :, :], cmap="bone", origin="lower")  # NCHW
    plt.title("MRI {}".format(idx), fontsize=20)

    plt.subplot(1, 3, 2)
    plt.imshow(msk[idx, 0, :, :], cmap="bone", origin="lower")
    plt.title("Ground truth", fontsize=20)

    # Predict using the OpenVINO model
    # NOTE: OpenVINO expects channels first for input and output
    input_img = img[[idx]]
    # print("input_img.shape: ", input_img.shape) ##(n,c,h,w) = (1,4,192,160)

    start_time = time.time()
    # Inferring using OpenVINO'S Sync API.
    prediction = exec_net([input_img])[output_layer]
    print("Elapsed time = {:.4f} msecs".format(
        1000.0*(time.time()-start_time)))

    plt.subplot(1, 3, 3)
    plt.imshow(prediction[0, 0, :, :], cmap="bone", origin="lower")
    dice_coef = calc_dice(msk[idx, :, :], prediction)
    plt.title("Prediction\nNew Dice = {:.4f}".format(dice_coef), fontsize=20)

    # Saving the predicted image
    if openvino_model_name == "FP32/unet2d":
        plt.savefig(
            '/home/Optimization/data_source/prediction_fp32.png',
            bbox_inches='tight')
    else:
        plt.savefig(
            '/home/Optimization/data_source/prediction_int8.png',
            bbox_inches='tight')


plot_results(ds_test)
plt.show()
