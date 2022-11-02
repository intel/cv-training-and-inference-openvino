## Structure of config.yaml :
Config file is divided into five sections.<br/>
* [pipeline](#pipeline)
* [download](#download)
* [preprocess](#preprocess)
* [train](#train)
* [convert](#convert)

## Pipeline: This is the key section of config file.
Here we define the list of tasks to be performed.
Possible tasks are { download, preprocess, train, convert }<br/>
User can skip the task by commenting or removing task from pipeline section.
Also make sure the task order is always maintained<br/>
**Correct Usage**   : [e.g. download->preprocess->train->convert]<br/>
```
pipeline:
        - download
        - preprocess
        - train: [fp32_8hpu_training]
        - convert: [onnx, pt]
```
**Incorrect Usage** : [e.g. preprocess->download->train->convert]</b><br/>
```
pipeline:
        - preprocess
        - download
        - train: [fp32_8hpu_training]
        - convert: [onnx, pt]
```
## Download:
Used for configuring download task parameters
For more details on download params, please refer [here](https://github.com/HabanaAI/Model-References/blob/6eb5cacf4c396a9eec7468c934c7d40eda00aa70/PyTorch/computer_vision/segmentation/Unet/download.py#L33)
```
download:
        task: '01'
        results: /data
```
## Preprocess - Used for configuring preprocess task parameters
For more details on preprocess params, please refer [here](https://github.com/HabanaAI/Model-References/blob/6eb5cacf4c396a9eec7468c934c7d40eda00aa70/PyTorch/computer_vision/segmentation/Unet/preprocess.py#L34)
Please note preprocess task creates a folder /data/01_2d for Unet2D dataset and feed this path for train data param.
```
preprocess:
        task: '01'
        dim: '2'
        data: /data
        results: /data
```
## Training - Defines training profiles and training params for each profiles.
User can define custom name for various training profiles which indicates the type of training<br/>
Each profile should be configured with own set of training params.
For more details on train params, please refer [here](https://github.com/HabanaAI/Model-References/blob/6eb5cacf4c396a9eec7468c934c7d40eda00aa70/PyTorch/computer_vision/segmentation/Unet/utils/utils.py#L223)
```
train:
    8hpu_train:
        data: /data/01_2d
        exec_mode: train
        hpus: '8'
        learning_rate: '0.0001'
    1hpu_train:
        data: /data/01_2d
        exec_mode: train
        hpus: '1'
```
## Convert - Defines output format types
This will be part of [pipeline](#pipeline-this-is-the-key-section-of-config-file) params itself. Please refer pipeline section for configuration.<br/>
Convert supports two formats: onnx (onnx format) and pt(pytorch model)
```
        - convert: [onnx, pt]
```

Final converted models will be stored directly under the training folder.<br/>
```
e.g.
.../gaudi-segmentation-unet-ptq/training$ ls *.onnx

unet2d.onnx
```

## Command to yaml format:
If you have habana training command and wanted to convert to yaml format, try running <font color="green">cmd2yaml.py</font> script.<br/>
Results can be copied to <font color="green">config/config.yaml</font> file to add a new training profile.
```
training/python$ python3 cmd2yaml.py --profile mixed_precision

Enter command string. Press CTRL-D to confirm
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 \
        --logname res_log --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_2d \
        --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 \
        --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --hmp \
        --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt \
        --deep_supervision --batch_size 64 --val_batch_size 64
train:
     mixed_precision:
         affinity: disabled
         batch_size: '64'
         data: /data/pytorch/unet/01_2d
         deep_supervision: ''
         dim: '2'
         exec_mode: train
         fold: '0'
         gpus: '0'
         hmp: ''
         hmp-bf16: ./config/ops_bf16_unet.txt
         hmp-fp32: ./config/ops_fp32_unet.txt
         hpus: '1'
         learning_rate: '0.001'
         logname: res_log
         norm: instance
         num_workers: '8'
         optimizer: fusedadamw
         results: /tmp/Unet/results/fold_0
         seed: '1'
         task: '01'
         val_batch_size: '64'
```
