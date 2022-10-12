'''
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'''

from models.nn_unet import NNUnet
import torch
import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_config():
    """
    Parses output format based on args
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--onnx", action="store_true", default=True,
                        help="stores output in onnx format")
    parser.add_argument("--pt", action="store_true", default=False,
                        help="stores output in pytorch format")
    args = parser.parse_args()
    return args


def main():
    """
    Main Function
    """
    args = get_config()
    ckpt_path = "/train/fold_0/checkpoints/last.ckpt"
    if not os.path.exists(ckpt_path):
        print("checkpoint not found to convert")
        exit(0)

    if args.pt or args.onnx:
        unet_model = NNUnet.load_from_checkpoint(ckpt_path)
        unet_model.eval()
        pt_unet = unet_model.model  # torch model

    if args.pt:
        torch.save(pt_unet, "torch_unet2d.pt")

    if args.onnx:
        X = torch.randn((64, 4, 192, 160))
        shape_dict = {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'}
        torch.onnx.export(
                        pt_unet,
                        X.to("cpu"),
                        "unet2d.onnx",
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': shape_dict,
                                      'output': shape_dict}
                        )


if __name__ == '__main__':
    main()
