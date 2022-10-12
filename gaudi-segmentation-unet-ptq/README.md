## An AWS cloud based generic AI Workflow which showcase model training on Habana® Gaudi® Processor using Amazon EC2 DL1 instance, followed by optimization and inference on  Intel® hardware using the OpenVINO™ toolkit.
This document contains instructions on how to run a model training and inference pipeline with Docker and Helm.

*	Introduction
    *	[Description](#Description)
    *	[Project Structure](#project-structure)
*	Get Started
    *	[Step 1: Training](#training)
    *	[Step 2: Inference](#inference)
* References
    *	[References](#references)

## Description
This repository contains the model scripts and recipe for training a U-Net 2D model to achieve state of the art accuracy using Image Segmentation with [Medical Decathlon](http://medicaldecathlon.com/) dataset and followed by inferencing with OpenVINO™ toolkit on  Intel® hardware. <br />

This AI workflow demonstartes the following: <br />
- U-Net 2D model training using [Amazon EC2 DL1 instances](https://aws.amazon.com/ec2/instance-types/dl1/) which uses Gaudi® Processor from Habana® Labs (an  Intel® company). <br />
- U-Net 2D model optimization and inference using OpenVINO™ toolkit on [Amazon M6i  Intel® CPU instances](https://aws.amazon.com/ec2/instance-types/m6i/) powered by 3rd Generation Intel® Xeon Scalable processors (code named Ice Lake). <br />

## Project Structure
```
├──training  - training related scripts and Helm chart
├──inference - optimization and inference related scripts and  Helm chart
├──README.md
```

## Training
[GoTo Training section](https://github.com/intel/cv-training-and-inference-openvino/tree/main/gaudi-segmentation-unet-ptq/training#training)

#### Option 1: Running training on 8 HPUs using Docker containers.
[GoTo Docker containers](https://github.com/intel/cv-training-and-inference-openvino/blob/main/gaudi-segmentation-unet-ptq/training#option-1-running-training-on-8-hpus-using-docker-containers)
#### Option 2: Running training on 8 HPUs with Helm Chart using Kubernetes.
[GoTo Helm chart](https://github.com/intel/cv-training-and-inference-openvino/blob/main/gaudi-segmentation-unet-ptq/training#option-2-running-training-on-8-hpus-with-helm-chart-using-kubernetes)

## Inference
[GoTo Inference section](https://github.com/intel/cv-training-and-inference-openvino/tree/main/gaudi-segmentation-unet-ptq/inference/onnx-to-ir-optimization#inference)
#### Option 1: Running optimization and inference using Docker containers.
[GoTo Docker containers](https://github.com/intel/cv-training-and-inference-openvino/blob/main/gaudi-segmentation-unet-ptq/inference/onnx-to-ir-optimization#option-1-running-optimization-and-inference-using-docker-containers)
#### Option 2: Running optimization and inference with Helm chart using Kubernetes.
[GoTo Helm chart](https://github.com/intel/cv-training-and-inference-openvino/blob/main/gaudi-segmentation-unet-ptq/inference/onnx-to-ir-optimization#option-2-running-optimization-and-inference-with-helm-chart-using-kubernetes)

## References
[HabanaAI/Model-References](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/computer_vision/segmentation/Unet) <br />
[Model used - NVIDIA's nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/2b20ca80cf7f08585e90a11c5b025fa42e4866c8/PyTorch/Segmentation/nnUNet). <br />
[IntelAI/unet](https://github.com/IntelAI/unet) <br />
[Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance) <br />
[developer.habana.ai](https://developer.habana.ai/resources) <br />
[OpenVINO™ documentation](https://docs.openvino.ai/latest/index.html) <br />
[OpenVINO™ toolkit download](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) <br />
