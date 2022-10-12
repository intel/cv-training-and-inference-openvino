#!/bin/bash
: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'

set -x
source paths.sh

# Clone Habana Model-Reference
echo "Cloning Model-References"
git clone https://github.com/HabanaAI/Model-References.git

# Preparing Docker image with required packages
if [ -z "$(docker image ls training_container -q)" ]; then
    echo "Preparing training_container"
    #pull habana/pytorch image
    docker pull $pytorch_gaudi_docker
    #assign habana_image as tag
    docker tag $pytorch_gaudi_docker habana_image
    #install unet requirements packages
    docker run \
        -v $train_path:/train \
        -v $model_path:/unet \
        --name habana_unet2d \
        habana_image \
        /train/scripts/requirements.sh
    # commit image as training_container
    docker commit habana_unet2d training_container
    # clear containers
    docker rm habana_unet2d
else
    echo "Skipping prepare training_container step"
fi
