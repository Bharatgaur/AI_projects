"""
Model Optimization: Pruning & Quantization
Terminal script (no UI). Trains, prunes, quantizes, then prints a comparison table.
Run: python main.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
import time
import tempfile

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
EPOCHS       = 3
BATCH_SIZE   = 64
LR           = 0.001
PRUNE_AMOUNT = 0.50   # 50% of weights pruned
DEVICE       = "cpu"

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def get_loaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = DataLoader(datasets.MNIST("./data", train=True,  download=True, transform=tf),
                       batch_size=BATCH_SIZE, shuffle=True)
    test  = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=tf),
                       batch_size=BATCH_SIZE)
    return train, test


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(32 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, 10)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ─────────────────────────────────────────────
# Train / Eval helpers
# ─────────────────────────────────────────────
def train(model, loader, opt, crit):
    model.train()
    loss_sum, correct = 0.0, 0
    for imgs, lbls in loader:
        opt.zero_grad()
        out  = model(imgs)
        loss = crit(out, lbls)
        loss.backward(); opt.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == lbls).sum().item()
    return loss_sum / len(loader.dataset), 100.0 * correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    loss_sum, correct = 0.0, 0
    for imgs, lbls in loader:
        out  = model(imgs)
        loss = crit(out, lbls)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == lbls).sum().item()
    return loss_sum / len(loader.dataset), 100.0 * correct / len(loader.dataset)


def model_size_kb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024
    os.remove(path)
    return size


def inference_ms(model, loader, n=10):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n: break
            t0 = time.perf_counter()
            model(imgs)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def sparsity(model):
    total = nz = 0
    for p in model.parameters():
        total += p.numel()
        nz    += p.nonzero().size(0)
    return 100.0 * (1 - nz / total)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" RTAI-242P | Practical 6 — Pruning & Quantization")
    print("=" * 60)

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    # ── 1. Train baseline ──────────────────────
    print("\n[1/3] Training baseline CNN …")
    model = SimpleCNN().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_loader, opt, criterion)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        print(f"  Epoch {ep}/{EPOCHS}  train_loss={tr_loss:.4f}  val_acc={va_acc:.2f}%")

    torch.save(model.state_dict(), "baseline_model.pt")
    base_acc    = va_acc
    base_size   = model_size_kb(model)
    base_infer  = inference_ms(model, test_loader)
    base_sparse = sparsity(model)

    # ── 2. Pruning ─────────────────────────────
    print(f"\n[2/3] Applying L1-Unstructured Pruning ({int(PRUNE_AMOUNT*100)}%) …")
    pruned = copy.deepcopy(model)
    for module, name in [(pruned.conv1, "weight"), (pruned.conv2, "weight"),
                         (pruned.fc1,   "weight"), (pruned.fc2,   "weight")]:
        prune.l1_unstructured(module, name=name, amount=PRUNE_AMOUNT)
        prune.remove(module, name)   # make permanent

    _, pruned_acc   = evaluate(pruned, test_loader, criterion)
    pruned_size     = model_size_kb(pruned)
    pruned_infer    = inference_ms(pruned, test_loader)
    pruned_sparse   = sparsity(pruned)
    torch.save(pruned.state_dict(), "pruned_model.pt")

    # ── 3. Quantization ────────────────────────
    print("[3/3] Applying Dynamic Post-Training Quantization (FP32 → INT8) …")
    quantized = torch.quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
    )
    _, quant_acc  = evaluate(quantized, test_loader, criterion)
    quant_size    = model_size_kb(quantized)
    quant_infer   = inference_ms(quantized, test_loader)
    quant_sparse  = sparsity(quantized)
    torch.save(quantized.state_dict(), "quantized_model.pt")

    # ── 4. Results table ──────────────────────
    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'Accuracy':>10} {'Size (KB)':>12} {'Inference (ms)':>16} {'Sparsity':>10}")
    print("-" * 70)
    for label, acc, size, infer, sp in [
        ("Baseline",  base_acc,   base_size,   base_infer,   base_sparse),
        ("Pruned",    pruned_acc, pruned_size,  pruned_infer, pruned_sparse),
        ("Quantized", quant_acc,  quant_size,   quant_infer,  quant_sparse),
    ]:
        print(f"{label:<15} {acc:>9.2f}% {size:>11.1f}  {infer:>15.2f}  {sp:>9.1f}%")
    print("=" * 70)

    print("\nAll models saved: baseline_model.pt | pruned_model.pt | quantized_model.pt")
    print("Done. Run `streamlit run app.py` for the interactive dashboard.\n")


if __name__ == "__main__":
    main()
