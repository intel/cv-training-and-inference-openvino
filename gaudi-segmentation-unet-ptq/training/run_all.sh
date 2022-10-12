#!/bin/bash
: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'

source paths.sh
# Clone Habana Model-Reference
echo "Cloning Model-References"
git clone https://github.com/HabanaAI/Model-References.git

set -x
# Preparing Docker image with required packages
if [ -z "$(docker image ls training_container -q)" ]; then
    echo "Preparing training_container"
    #pull habana/pytorch image
    docker pull $pytorch_gaudi_docker
    #assign habana_image as tag
    docker tag $pytorch_gaudi_docker habana_image
    #install unet requirements packages
    docker run -u root \
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

# Start pipeline processes configured in config.yaml
docker run -u root \
    -v $train_path:/train \
    -v $model_path:/unet \
    -v $data_path:/data \
    -v $(pwd)/fold_0:/tmp/Unet/results/fold_0 \
    training_container \
    /train/scripts/config_run.sh
