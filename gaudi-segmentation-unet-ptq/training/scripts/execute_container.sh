#!/bin/bash
: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'

set -x
source paths.sh

# Start pipeline processes configured in config.yaml
docker run -u root \
    -v $train_path:/train \
    -v $model_path:/unet \
    -v $data_path:/data \
    -v $(pwd)/fold_0:/tmp/Unet/results/fold_0 \
    training_container \
    /train/scripts/config_run.sh
