#!/bin/bash

: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'

bucket_name=$1
zip_file_name=$2
output_path=$3

if [ ! -z $bucket_name ] && [ ! -z $zip_file_name ] && [ ! -z $output_path ];
then
 aws s3 cp "s3://$bucket_name/models/$zip_file_name" $output_path
 download_status=$?
 if [ $download_status -eq 0 ];
 then
  echo "Download, Successful"
  unzip  "$output_path/$zip_file_name" -d "$output_path"
  rm -rf "$output_path/$zip_file_name"
 else
  echo "Download, Failed."
 fi
else
 echo "Please make sure, all arguments are passed. Required arguments are 'bucket_name', 'zip_file_name', 'output_path'"
fi
