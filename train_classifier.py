# core/train_classifier.py
"""
Train the SmallCNN classifier on an ImageFolder-structured dataset.

Usage:
python core/train_classifier.py --data_dir data/synth --epochs 10 --batch_size 128 --out_dir models
"""

import os
import argparse
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SmallCNN

CLASS_MAP = [
    "empty",
    "wP","wN","wB","wR","wQ","wK",
    "bP","bN","bB","bR","bQ","bK"
]

def make_dataloaders(data_dir, img_size=64, batch_size=128, val_split=0.1, seed=42):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    full = datasets.ImageFolder(data_dir, transform=transform_train)
    # split
    n = len(full)
    idx = list(range(n))
    split = int(n * (1 - val_split))
    import random
    random.seed(seed); random.shuffle(idx)
    train_idx, val_idx = idx[:split], idx[split:]
    from torch.utils.data import Subset
    train_ds = Subset(full, train_idx)
    # for val use same dataset class but with val transforms
    full_val = datasets.ImageFolder(data_dir, transform=transform_val)
    val_ds = Subset(full_val, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, full.class_to_idx

def train(data_dir, out_dir="models", epochs=10, batch_size=128, lr=1e-3, img_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    train_loader, val_loader, class_to_idx = make_dataloaders(data_dir, img_size, batch_size)
    num_classes = len(class_to_idx)
    model = SmallCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        acc = correct / total if total > 0 else 0.0
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f} time={time.time()-t0:.1f}s")
        # checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(out_dir, "best_state_dict.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("Saved best state_dict to", ckpt_path)
    # save final model and torchscript
    final_path = os.path.join(out_dir, "final_state_dict.pth")
    torch.save(model.state_dict(), final_path)
    print("Saved final state_dict to", final_path)

    # create a scripted model (trace)
    model.eval()
    example = torch.randn(1, 3, img_size, img_size).to(device)
    try:
        traced = torch.jit.trace(model.cpu(), example.cpu())
        ts_path = os.path.join(out_dir, "model_scripted.pt")
        traced.save(ts_path)
        print("Saved TorchScript model to", ts_path)
    except Exception as e:
        print("TorchScript tracing failed:", e)

    # also save a simple metadata file mapping class indices to folder names
    meta = {"class_to_idx": class_to_idx}
    import json
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Training complete. Models in", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/synth")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()
    train(args.data_dir, args.out_dir, args.epochs, args.batch_size, args.lr, args.img_size)
