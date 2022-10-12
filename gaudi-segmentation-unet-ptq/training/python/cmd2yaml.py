'''
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'''

import yaml
import re
import sys
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def parse_args():
    """
    Parses profile name from args
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--profile", type=str, required=True,
                        help="Provide profile used in config yaml")
    args = parser.parse_args()
    return args


def get_command():
    """
    Get command string from user
    """
    print(" ".join(sys.argv))
    print("Enter command string and Press CTRL-D to confirm")
    cmd_str = " ".join(sys.stdin.readlines())
    cmd_str = cmd_str[cmd_str.find(".py")+len(".py"):]
    print(cmd_str)
    return cmd_str


def cmd_to_args():
    """
    Extract key-values from command string
    """
    regexp = re.compile(r'([--]{2}|[-]{1})([a-z,A-Z,0-9,\-,_]+)[ ]*'
                        r'(?!--|-)([a-z,A-Z,0-9,\-,_,/,\.]*)')
    cmd_str = get_command()
    kv_args = []

    for val in re.findall(regexp, cmd_str):
        kv_args.append(val)

    return kv_args


def main():
    """
    Main Function
    """
    args = parse_args()
    kv_args = cmd_to_args()
    new_profile = {}
        yml_dict = {'train': {}}

    yml_dict['train'][args.profile] = new_profile
    for arg in kv_args:
        new_profile[arg[1]] = arg[2]

    tmp_config = './temp_config.yaml'

    with open(tmp_config, 'w') as f:
        yaml.safe_dump(yml_dict, f, width=70, indent=4)

    with open(tmp_config, 'r') as f:
        for line in f.readlines():
            print(line)

    os.remove(tmp_config)


if __name__ == '__main__':
    main()
