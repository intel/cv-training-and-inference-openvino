#!/bin/bash
: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'

export PYTHON=$(which python3)
echo $PYTHON
export $PYTHONPATH:/unet
echo $PYTHONPATH
$PYTHON -m pip install --upgrade pip
pip install -r /unet/requirements.txt
