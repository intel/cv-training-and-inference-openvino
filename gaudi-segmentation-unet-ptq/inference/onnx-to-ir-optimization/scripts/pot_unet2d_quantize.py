
'''
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'''

# Refer:
# https://docs.openvino.ai/latest/notebooks/111-detection-quantization-with-output.html

import os
import numpy as np
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
# from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline

import preprocess
import nibabel as nib

'''
# Modify the Data Loader implementation depending on your dataset and
# model's pre-processing requirement's.
'''


class UnetImageLoader(DataLoader):
    def __init__(self, dataset_path):
        self._train_img_files = []
        self._label_files = []

        self.patch_size = [192, 160]
        dp_str = ""
        dp_str.join(dataset_path)
        train_images_path = dataset_path + dp_str + "imagesTr"
        print("train_images path: ", train_images_path)
        labels_path = dataset_path + dp_str + "labelsTr"
        print("labels_path: ", labels_path)

        for file_name in os.listdir(train_images_path):
            f = os.path.join(train_images_path, file_name)
            self._train_img_files.append(f)

        for file_name in os.listdir(labels_path):
            f = os.path.join(labels_path, file_name)
            self._label_files.append(f)

    def __len__(self):
        return len(self._train_img_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image_path = self._train_img_files[index]
        label_path = self._label_files[index]

        img = np.array(nib.load(image_path).dataobj)
        label = np.array(nib.load(label_path).dataobj)

        # Preprocess input and label
        img, label = preprocess.apply(img, label, self.patch_size)
        img = np.transpose(img, [1, 0, 2, 3])  # depth,channel,height,width
        num_slices = img.shape[0]
        slice_idx = np.random.choice(range(num_slices))
        img = img[[slice_idx]]
        preprocessed_image = img
        '''
        print(
            "pre-processed image shape: ", preprocessed_image.shape
        )  # n,c,h,w
        '''

        return preprocessed_image, None


def run_pot_unet2d(dataset_path, ir_data, output_dir, out_model_name):
    print('Running quantization for UNet2D model '
          'using POT DefaultQuantization Algorithm.')

    # Model config specifies the model name and
    #  paths to model .xml and .bin file

    # location of FP32 IR to be quantized
    path_to_xml = ir_data[0]
    path_to_bin = ir_data[1]
    model_config = {
        "model_name": "unet_2d",
        "model": path_to_xml,
        "weights": path_to_bin,
    }

    # Engine config
    engine_config = {"device": "CPU"}

    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "preset": "mixed",
                "target_device": "CPU",
                "stat_subset_size": 300,
            },

        }
    ]

    # Step 1: Implement and create user's data loader
    data_loader = UnetImageLoader(
        dataset_path=dataset_path)  # other img folder

    # Step 2: Load model
    model = load_model(model_config=model_config)

    # Step 3: Initialize the engine for
    #          metric calculation and statistics collection.
    engine = IEEngine(config=engine_config, data_loader=data_loader)

    # Step 4: Create a pipeline of compression algorithms and run it.
    pipeline = create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(model=model)

    # Step 5 (Optional): Compress model weights to quantized precision
    #                     to reduce the size of the final .bin file.
    # compress_model_weights(compressed_model)

    # Step 6: Save the compressed model to the desired path.
    # Set save_path to the directory where the model should be saved
    print("Saving quantized model...")
    save_model(
        model=compressed_model,
        save_path=output_dir,
        model_name=out_model_name,
    )
    print("Done.\n")


if __name__ == "__main__":

    DATASET_PATH = "/home/Optimization/data_source/Task01_BrainTumour/"

    IR_XML_BIN_TO_QUANTIZE = ("/home/Optimization/ir_model/FP32/unet2d.xml",
                              "/home/Optimization/ir_model/FP32/unet2d.bin")
    OUT_INT8IR_DIR = "/home/Optimization/ir_model/INT8"
    OUT_MODEL_NAME = "unet2d_int8"
    run_pot_unet2d(DATASET_PATH, IR_XML_BIN_TO_QUANTIZE,
                   OUT_INT8IR_DIR, OUT_MODEL_NAME)
