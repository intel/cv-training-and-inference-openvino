#!/bin/bash
: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'

# Provide path to training folder
# by default assumes current dir as training
train_path=$(pwd)
# Provide path to which dataset/pre-processed outputs should be stored
# by default creates data folder under training path
# it is recommended to always provide new path
data_path=$train_path/data

# DON'T CHANGE THE FOLLOWING PATH UNLESS IT IS NECESSARY
model_path=$train_path/Model-References/PyTorch/computer_vision/segmentation/Unet
pytorch_gaudi_docker="vault.habana.ai/gaudi-docker/1.6.1/ubuntu20.04/habanalabs/pytorch-installer-1.12.0:latest"

function validate(){

  # Validate training
  if [ ! -d $train_path ]; then
    echo -e "ERROR Loading Training Path:\n  $train_path not exists"
    exit 1
  fi

  # Validate data path
  if [ -d $data_path ]; then
    echo -e "ERROR Data path already exists:\n $data_path\n Please provide new path"
    exit 1
  else
    echo -e "Configured Data Path:\n  $data_path"
  fi

  # No validation required as this path gets created later
  echo -e "Configured Model Path:\n  $model_path"
}

validate
