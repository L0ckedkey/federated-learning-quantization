import torch
import torch.nn as nn
import numpy as np
import onnx
import os
from models.base import ECGCNN
from config import *
from utils.train import get_general_model_size

DEVICE = torch.device("cuda")
input_length = 28
num_classes = 5

model = ECGCNN(input_channels=1, num_classes=num_classes, input_length=input_length).to(DEVICE)
model.load_state_dict(torch.load("", map_location=DEVICE))
model.eval()

print("✅ PyTorch model loaded.")

dummy_input = torch.randn(1, input_length, 1).to(DEVICE) 

onnx_path = os.path.join("./", ".onnx")

torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print(f"✅ Model exported to {onnx_path}")

get_general_model_size(onnx_path)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model checked successfully.")