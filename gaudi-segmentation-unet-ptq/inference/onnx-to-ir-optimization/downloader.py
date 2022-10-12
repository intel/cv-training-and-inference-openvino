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
PyTorch/computer_vision/segmentation/Unet/download.py

--------------------------------------------------------------------------
'''

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import call
import shlex

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

if __name__ == "__main__":
    args = parser.parse_args()
    tar_file = "Task01_BrainTumour.tar"
    dataset_file = "Task01_BrainTumour/"
    # Path to /data_source
    file_path = os.path.join(os.getcwd(), "data_source")

    # Path to /Task01_BrainTumour.tar
    tar_file_path = file_path + "/" + tar_file

    # Path to /Task01_BrainTumour
    dataset_path = file_path + "/" + dataset_file

    # Check whether the dataset_path path already exists.
    is_dataset_path = os.path.isdir(dataset_path)
    if is_dataset_path:
        print(f"Dataset folder - 'Task01_BrainTumour' "
              f"already exists at: {dataset_path}")
        print(f"If you want to download it again, "
              f"delete the directory 'Task01_BrainTumour' from:{file_path}  "
              f"and run the script again.")
        exit()

    # Path to Train, Test and label directories from the dataset
    imagesTr_path = os.path.join(dataset_path, "imagesTr/._BRATS_*")
    imagesTs_path = os.path.join(dataset_path, "imagesTs/._BRATS_*")
    labelsTr_path = os.path.join(dataset_path, "labelsTr/._BRATS_*")

    # Downloads the dataset
    call(
        shlex.split(f"aws s3 cp s3://msd-for-monai-eu/{tar_file} \
        --no-sign-request {file_path}"),
        shell=False
    )

    # Untar the downloaded Task01_BrainTumour.tar file
    call(shlex.split(f"tar -xvf {tar_file_path} -C {file_path} "), shell=False)

    # Remove unwanted files from the downloaded dataset folder
    call(shlex.split(f"rm -rf {imagesTr_path}"), shell=False)
    call(shlex.split(f"rm -rf {imagesTs_path}"), shell=False)
    call(shlex.split(f"rm -rf {labelsTr_path}"), shell=False)

    # Remove the Task01_BrainTumour.tar file
    call(shlex.split(f"rm -rf {tar_file_path}"), shell=False)
