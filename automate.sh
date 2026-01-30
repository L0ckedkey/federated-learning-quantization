#!/bin/bash

echo "Menjalankan FL Client"

source ~/jp6/bin/activate
cd  /home/hans/Documents/Thesis && python ./fl/client.py -c jetson1_preprocess_new_svdb_path -d SVDB_PATH

echo "FL Client Done !"

echo "Activate TensorRT Environment"

python onnx-tflite.py
source ~/trt/bin/activate

echo "TensorRT environment on"

echo "Proses Quantization TensorRT"

# 3. Proses quantisasi (contoh)
trtexec --onnx=./model.onnx --saveEngine=nama_file_fp16.plan --fp16
trtexec --onnx=path_onnx_model         --int8         --saveEngine=nama_file_int8.engine         --calib=path_calib_data         --optShapes=input:model_specs         --minShapes=input:model_specs         --maxShapes=input:model_specs
echo "Done"
