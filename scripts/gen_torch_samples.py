#!/usr/bin/env python3
"""Export a few real-world style models from PyTorch (random weights).

value_info is stripped from the saved files so the app has to infer every
intermediate shape itself; ONNX's reference inference is stored next to each
model as <name>.shapes.json.
"""
import argparse
import json
import os

import onnx
import torch
import torchvision
from torch import nn

from gen_samples import ref_shapes


class Block(nn.Module):
    def __init__(self, d=64, heads=4, ff=128):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, d // self.heads).transpose(1, 2)
        k = k.view(b, s, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, s, self.heads, d // self.heads).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / (d // self.heads) ** 0.5, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.head = nn.Linear(64, 1000)

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y)


def export(model, args, name, out, index, desc, dynamic_axes, input_names, output_names, opset=17):
    path = os.path.join(out, name + ".onnx")
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    m = onnx.load(path)
    shapes = ref_shapes(m)
    del m.graph.value_info[:]
    onnx.save(m, path)
    if shapes is not None:
        with open(os.path.join(out, name + ".shapes.json"), "w") as f:
            json.dump(shapes, f, indent=1)
    index.append({"name": name, "file": name + ".onnx", "description": desc})
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    index = []
    export(
        torchvision.models.resnet18(weights=None),
        (torch.randn(1, 3, 224, 224),),
        "resnet18",
        a.out,
        index,
        "torchvision ResNet-18 (random weights), dynamic batch",
        {"input": {0: "batch"}, "output": {0: "batch"}},
        ["input"],
        ["output"],
    )
    export(
        torchvision.models.mobilenet_v3_small(weights=None),
        (torch.randn(1, 3, 224, 224),),
        "mobilenet_v3_small",
        a.out,
        index,
        "torchvision MobileNetV3-Small (random weights)",
        {"input": {0: "batch"}, "output": {0: "batch"}},
        ["input"],
        ["output"],
    )
    export(
        TinyTransformer(),
        (torch.randint(0, 1000, (2, 16)),),
        "tiny_transformer",
        a.out,
        index,
        "2-layer transformer exported by PyTorch, dynamic batch and sequence",
        {"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        ["input_ids"],
        ["logits"],
    )
    export(
        BiLSTM(),
        (torch.randn(2, 10, 32),),
        "bilstm",
        a.out,
        index,
        "2-layer bidirectional LSTM",
        {"x": {0: "batch", 1: "time"}, "y": {0: "batch", 1: "time"}},
        ["x"],
        ["y"],
    )
    index_path = os.path.join(a.out, "index.json")
    existing = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
    names = {e["name"] for e in index}
    with open(index_path, "w") as f:
        json.dump([e for e in existing if e["name"] not in names] + index, f, indent=1)


if __name__ == "__main__":
    main()
