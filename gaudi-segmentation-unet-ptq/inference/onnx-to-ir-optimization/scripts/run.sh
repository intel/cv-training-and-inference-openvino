#!/bin/bash

: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'


# Installing Required Packages for POT and OpenVINO™ Inference
python3 -m pip install --upgrade pip
pip install matplotlib
pip install psutil
pip install monai

OPTIMIZATION="${OPTIMIZATION:- false}"
if [ "$OPTIMIZATION" == true ]; 
then
    echo "Started Quantizing the model"
    # Running Post Training Optimization on the FP32 IR Files
    python3 /home/Optimization/scripts/pot_unet2d_quantize.py
elif [ "$OPTIMIZATION" == false ] && [ "$PRECISION" = "FP32" ];
then
    echo "Started executing the inference sample script with FP32 model"
    # Running Inference on the FP32 IR files
    python3 /home/Optimization/scripts/inference.py --ov_model_name 'FP32/unet2d'
elif [ "$OPTIMIZATION" == false ] && [ "$PRECISION" = "INT8" ];
then
    echo "Started executing the inference sample script with INT8 model"
    # Running Inference on the INT8 IR files
    python3 /home/Optimization/scripts/inference.py --ov_model_name 'INT8/unet2d_int8'
else
    echo "Please assign the right value to --env setting when running the container"
fi
exit
