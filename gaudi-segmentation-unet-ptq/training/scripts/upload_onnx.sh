#!/bin/bash
: '
-------------------------------------------------------------------------
# Copyright(C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
--------------------------------------------------------------------------
'

bucket_name=$1
zip_file_name=$2
model_path=$3

if [ ! -z $bucket_name ] && [ ! -z $zip_file_name ] && [ ! -z $model_path ];
then
 OUTPUT_ZIP_FILE_PATH="$(pwd)/$zip_file_name"

 zip -r $OUTPUT_ZIP_FILE_PATH  $model_path
 
 zip_status=$?
 if [ $zip_status -eq 0 ];
 then
  aws s3 cp $OUTPUT_ZIP_FILE_PATH "s3://$bucket_name/models/"
  upload_status=$?
  if [ $upload_status -eq 0 ];
  then
    echo "Upload, Successful."
    rm -rf $OUTPUT_ZIP_FILE_PATH
    exit
  else
   echo "Upload, Failed."
   exit
  fi
 else
  echo "Unable to ZIP"
  exit
 fi
else
 echo "Please make sure, all arguments are passed. Required arguments are 'bucket_name', 'zip_file_name', 'model_path'"
 exit
fi