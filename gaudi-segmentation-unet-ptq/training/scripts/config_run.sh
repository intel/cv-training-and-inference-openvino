#!/bin/bash
: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'
export PYTHONPATH=$PYTHONPATH:/unet
cd /train
python3 python/run_config.py
