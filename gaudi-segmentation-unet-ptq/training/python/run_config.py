'''
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'''

import yaml
import os
import time
import shlex
import subprocess

log_path = '/train/logs'


def read_config():
    """
    Reads config/config.yaml file and returns config as python dict
    """
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        print("config not found")
        exit(0)

    with open(config_path, 'r') as fp_yml:
        config_dict = yaml.safe_load(fp_yml)
    return config_dict


def cmd_params(params):
    """
    Converts params dict to command-line string
    """
    return " ".join([f"--{k} {v}" if k != ''
                     else f"--{k}"
                     for k, v in params.items()])


def exec_cmd(command, log_file):
    """
    Executes command and redirects output to log_file
    """
    cmd = subprocess.Popen(shlex.split(command), shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    tee = subprocess.Popen(shlex.split(f"tee {log_file}"), shell=False,
                           stdin=cmd.stdout)
    tee.wait()


def run_download(download_args):
    """
    Triggers dataset download script
    """
    os.chdir('/unet')
    download_log = f"{log_path}/download_"\
                   f"{time.strftime('%Y-%d-%m__%Hh%Mm%Ss.log')}"
    command = "python3 download.py " + cmd_params(download_args)
    exec_cmd(command, download_log)


def run_preprocess(preprocess_args):
    """
    Triggers data preprocess script
    """
    os.chdir('/unet')
    preprocess_log = f"{log_path}/preprocess_"\
                     f"{time.strftime('%Y-%d-%m__%Hh%Mm%Ss.log')}"
    command = "python3 preprocess.py " \
              + cmd_params(preprocess_args)
    exec_cmd(command, preprocess_log)


def run_convert(convert_args):
    """
    Triggers model conversion script
    """
    os.chdir('/train')
    convert_args = " ".join([f"--{k}" for k in convert_args])
    convert_log = f"{log_path}/convert_"\
                  f"{time.strftime('%Y-%d-%m__%Hh%Mm%Ss.log')}"
    command = "python3 /train/python/get_output.py "\
              + convert_args
    exec_cmd(command, convert_log)


def run_train(train_args):
    """
    Triggers habana training script
    """
    os.chdir('/unet')
    training_log = f"{log_path}/training_"\
                   f"{time.strftime('%Y-%d-%m__%Hh%Mm%Ss.log')}"
    command = "python3 -u main.py " + cmd_params(train_args)
    exec_cmd(command, training_log)


def main():
    """
    Main Function
    """
    download_args = None
    preprocess_args = None
    convert_args = None

    conf = read_config()
    os.makedirs(log_path, exist_ok=True)

    for v in conf['pipeline']:
        if type(v) == str:
            if v == 'download':
                download_args = conf[v]
                run_download(download_args)
            elif v == 'preprocess':
                preprocess_args = conf[v]
                run_preprocess(preprocess_args)

        elif type(v) == dict:
            for k, p in v.items():
                if k == 'train':
                    for prof in p:
                        train_params = conf[k][prof]
                        if train_params.get('save_ckpt', False) != '':
                            train_params['save_ckpt'] = ''
                        run_train(train_params)
                elif k == 'convert':
                    convert_args = p
                    run_convert(convert_args)

            print(v.items())


if __name__ == "__main__":
    main()
