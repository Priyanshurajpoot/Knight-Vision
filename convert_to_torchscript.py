# core/convert_to_torchscript.py
import torch
from models import SmallCNN
import argparse
import os

def convert(state_dict_path, out_path="model_scripted.pt", img_size=64, device="cpu"):
    sd = torch.load(state_dict_path, map_location=device)
    # adjust num_classes if state has different FC size
    # Here we assume the SmallCNN architecture used in training
    num_classes = sd.get("fc.weight", None)
    model = SmallCNN()
    model.load_state_dict(sd)
    model.eval()
    example = torch.randn(1,3,img_size,img_size)
    traced = torch.jit.trace(model, example)
    traced.save(out_path)
    print("Saved TorchScript model to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", default="model_scripted.pt")
    args = parser.parse_args()
    convert(args.state, args.out)
