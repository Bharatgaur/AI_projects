"""
Model Optimization — Pruning & Quantization
Streamlit UI to train, prune, quantize, and compare a CNN on MNIST.
Run: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import copy
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Model Optimization — Pruning & Quantization",
    page_icon="⚡",
    layout="wide",
)

st.title("Model Optimization: Pruning & Quantization")
st.markdown(
    "> Reduce model size & improve inference speed using Pruning + Post-Training Quantization (PyTorch)"
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
DEVICE = "cpu"   # Keep on CPU so quantization works everywhere


def get_dataloaders(batch_size=64):
    """Download MNIST and return train / test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ──────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """Lightweight CNN for MNIST (fast enough for classroom demos)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28→14
        x = self.pool(self.relu(self.conv2(x)))   # 14→7
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ──────────────────────────────────────────────
# Training / Evaluation utilities
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


def count_parameters(model):
    """Total & non-zero parameters."""
    total   = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    return total, nonzero


def model_size_kb(model):
    """Size on disk in KB by saving to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024
    os.remove(path)
    return size


def measure_inference_ms(model, loader, n_batches=10):
    """Average inference time (ms) per batch."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            t0 = time.perf_counter()
            _ = model(images.to(DEVICE))
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


# ──────────────────────────────────────────────
# Sidebar — Hyperparameters
# ──────────────────────────────────────────────
st.sidebar.header("Hyperparameters")
epochs       = st.sidebar.slider("Training Epochs",       1, 10, 3)
batch_size   = st.sidebar.selectbox("Batch Size",         [32, 64, 128], index=1)
lr           = st.sidebar.select_slider("Learning Rate",  [0.0001, 0.001, 0.01], value=0.001)
prune_amount = st.sidebar.slider("Pruning Amount (%)",    10, 90, 50) / 100
st.sidebar.markdown("---")
st.sidebar.info(
    "**Steps performed:**\n"
    "1. Train baseline model\n"
    "2. Apply L1-Unstructured Pruning\n"
    "3. Apply Dynamic Quantization\n"
    "4. Compare size, accuracy & speed"
)

# ──────────────────────────────────────────────
# Session state — persist models across reruns
# ──────────────────────────────────────────────
for key in ["baseline", "pruned", "quantized", "results", "train_history"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ──────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    " Train Baseline",
    " Pruning",
    " Quantization",
    " Comparison Dashboard",
])

# ══════════════════════════════════════════════
# TAB 1 — Train Baseline
# ══════════════════════════════════════════════
with tab1:
    st.header("Step 1 — Train Baseline CNN on MNIST")
    st.markdown(
        "We first train a small CNN (2 conv layers + 2 FC layers) on MNIST from scratch. "
        "This **baseline** model will be our reference for size and accuracy."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Architecture")
        st.code(
            "SimpleCNN(\n"
            "  Conv2d(1→16, 3×3)  + ReLU + MaxPool\n"
            "  Conv2d(16→32, 3×3) + ReLU + MaxPool\n"
            "  Linear(1568→128)   + ReLU + Dropout\n"
            "  Linear(128→10)     [output]\n"
            ")",
            language="text",
        )
    with col2:
        st.subheader("Dataset")
        st.markdown(
            "- **MNIST** handwritten digits (auto-downloaded)\n"
            "- 60,000 train / 10,000 test images\n"
            "- Normalised: mean=0.1307, std=0.3081"
        )

    if st.button(" Train Baseline Model", type="primary"):
        with st.spinner("Downloading MNIST & training…"):
            train_loader, test_loader = get_dataloaders(batch_size)
            model     = SimpleCNN().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
            prog    = st.progress(0, text="Training…")

            for ep in range(1, epochs + 1):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
                va_loss, va_acc = evaluate(model, test_loader, criterion)
                history["train_acc"].append(tr_acc)
                history["val_acc"].append(va_acc)
                history["train_loss"].append(tr_loss)
                history["val_loss"].append(va_loss)
                prog.progress(ep / epochs, text=f"Epoch {ep}/{epochs} — Val Acc: {va_acc:.2f}%")

            st.session_state.baseline     = model
            st.session_state.train_history = history
            torch.save(model.state_dict(), "baseline_model.pt")

        st.success(f" Training complete! Final Val Accuracy: **{va_acc:.2f}%**")

        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        eps = range(1, epochs + 1)
        axes[0].plot(eps, history["train_acc"], "b-o", label="Train")
        axes[0].plot(eps, history["val_acc"],   "r-o", label="Val")
        axes[0].set_title("Accuracy per Epoch"); axes[0].legend(); axes[0].set_xlabel("Epoch")
        axes[1].plot(eps, history["train_loss"], "b-o", label="Train")
        axes[1].plot(eps, history["val_loss"],   "r-o", label="Val")
        axes[1].set_title("Loss per Epoch"); axes[1].legend(); axes[1].set_xlabel("Epoch")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        total, nz = count_parameters(model)
        size_kb   = model_size_kb(model)
        infer_ms  = measure_inference_ms(model, test_loader)

        st.info(
            f"**Baseline Stats:**  \n"
            f"• Total params: **{total:,}**  \n"
            f"• Model size: **{size_kb:.1f} KB**  \n"
            f"• Avg inference: **{infer_ms:.2f} ms/batch**"
        )

    elif st.session_state.baseline is not None:
        st.success(" Baseline model already trained. Proceed to pruning →")


# ══════════════════════════════════════════════
# TAB 2 — Pruning
# ══════════════════════════════════════════════
with tab2:
    st.header("Step 2 — L1-Unstructured Pruning")
    st.markdown(
        "**Pruning** removes weights with the smallest absolute values, forcing them to zero. "
        "This reduces the *effective* number of parameters without changing the architecture."
    )

    st.subheader("How Pruning Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**L1-Unstructured Pruning:**\n"
            "- Ranks all weights by |magnitude|\n"
            "- Sets bottom X% to **zero** (via a mask)\n"
            "- Works on Conv & Linear layers\n"
            "- `make_permanent()` bakes masks into weights"
        )
    with col2:
        st.code(
            "import torch.nn.utils.prune as prune\n\n"
            "# Prune 50% of weights in a layer\n"
            "prune.l1_unstructured(\n"
            "    module=model.fc1,\n"
            "    name='weight',\n"
            "    amount=0.50\n"
            ")\n"
            "# Make permanent\n"
            "prune.remove(model.fc1, 'weight')",
            language="python",
        )

    if st.button(" Apply Pruning", type="primary"):
        if st.session_state.baseline is None:
            st.error(" Train the baseline model first!")
        else:
            _, test_loader = get_dataloaders(batch_size)
            criterion      = nn.CrossEntropyLoss()

            # Deep-copy so baseline is unchanged
            pruned_model = copy.deepcopy(st.session_state.baseline)

            # Apply L1 unstructured pruning to all Conv and Linear layers
            layers_to_prune = [
                (pruned_model.conv1, "weight"),
                (pruned_model.conv2, "weight"),
                (pruned_model.fc1,   "weight"),
                (pruned_model.fc2,   "weight"),
            ]
            for module, name in layers_to_prune:
                prune.l1_unstructured(module, name=name, amount=prune_amount)
                prune.remove(module, name)   # make permanent

            _, pruned_acc = evaluate(pruned_model, test_loader, criterion)
            total, nz     = count_parameters(pruned_model)
            sparsity      = 100.0 * (1 - nz / total)
            size_kb       = model_size_kb(pruned_model)
            infer_ms      = measure_inference_ms(pruned_model, test_loader)

            st.session_state.pruned = pruned_model
            torch.save(pruned_model.state_dict(), "pruned_model.pt")

            # Baseline stats
            _, base_loader = get_dataloaders(batch_size)
            _, base_acc    = evaluate(st.session_state.baseline, test_loader, criterion)
            base_total, _  = count_parameters(st.session_state.baseline)
            base_size      = model_size_kb(st.session_state.baseline)
            base_infer     = measure_inference_ms(st.session_state.baseline, test_loader)

            st.success(f" Pruning done! Sparsity achieved: **{sparsity:.1f}%**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",       f"{pruned_acc:.2f}%",   f"{pruned_acc - base_acc:+.2f}%")
            c2.metric("Non-zero Params", f"{nz:,}",              f"{nz - base_total:,}")
            c3.metric("Model Size",      f"{size_kb:.1f} KB",   f"{size_kb - base_size:+.1f} KB")
            c4.metric("Inference",       f"{infer_ms:.2f} ms",  f"{infer_ms - base_infer:+.2f} ms")

            # Visualise sparsity in fc1
            fig, ax = plt.subplots(figsize=(8, 2))
            w = pruned_model.fc1.weight.detach().numpy().flatten()
            ax.hist(w, bins=100, color="steelblue", edgecolor="white")
            ax.axvline(0, color="red", linewidth=1.5, label="Zero (pruned)")
            ax.set_title(f"Weight Distribution of fc1 after {int(prune_amount*100)}% pruning")
            ax.legend(); ax.set_xlabel("Weight value")
            st.pyplot(fig); plt.close(fig)

    elif st.session_state.pruned is not None:
        st.success(" Pruned model ready. Proceed to quantization →")


# ══════════════════════════════════════════════
# TAB 3 — Quantization
# ══════════════════════════════════════════════
with tab3:
    st.header("Step 3 — Dynamic Post-Training Quantization")
    st.markdown(
        "**Quantization** reduces weight precision from **FP32 → INT8**, cutting model size ~4× "
        "and speeding up inference on CPU."
    )

    st.subheader("Types of Quantization")
    st.table({
        "Type"           : ["Post-Training Dynamic (used here)", "Post-Training Static", "Quantization-Aware Training"],
        "Calibration"    : ["None needed",  "Needs calibration dataset", "Retrain with fake quant"],
        "Accuracy Loss"  : ["Minimal",      "Very small",                "Lowest"],
        "Ease of Use"    : ["⭐⭐⭐ Easiest", "⭐⭐ Medium",               "⭐ Harder"],
    })

    st.code(
        "import torch\n\n"
        "# Apply dynamic quantization — converts Linear layers to INT8\n"
        "quantized_model = torch.quantization.quantize_dynamic(\n"
        "    model,\n"
        "    {torch.nn.Linear},   # layers to quantize\n"
        "    dtype=torch.qint8\n"
        ")",
        language="python",
    )

    if st.button(" Apply Quantization", type="primary"):
        if st.session_state.baseline is None:
            st.error("Train the baseline model first!")
        else:
            _, test_loader = get_dataloaders(batch_size)
            criterion      = nn.CrossEntropyLoss()

            q_model = torch.quantization.quantize_dynamic(
                copy.deepcopy(st.session_state.baseline),
                {nn.Linear},
                dtype=torch.qint8,
            )

            _, q_acc    = evaluate(q_model, test_loader, criterion)
            _, base_acc = evaluate(st.session_state.baseline, test_loader, criterion)
            q_size      = model_size_kb(q_model)
            base_size   = model_size_kb(st.session_state.baseline)
            q_infer     = measure_inference_ms(q_model, test_loader)
            base_infer  = measure_inference_ms(st.session_state.baseline, test_loader)

            st.session_state.quantized = q_model
            torch.save(q_model.state_dict(), "quantized_model.pt")

            st.success("Quantization applied!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",     f"{q_acc:.2f}%",    f"{q_acc - base_acc:+.2f}%")
            c2.metric("Model Size",   f"{q_size:.1f} KB", f"{q_size - base_size:+.1f} KB")
            c3.metric("Size Reduction", f"{(1 - q_size/base_size)*100:.1f}%", "vs baseline")
            c4.metric("Inference",    f"{q_infer:.2f} ms", f"{q_infer - base_infer:+.2f} ms")

    elif st.session_state.quantized is not None:
        st.success("Quantized model ready. Check the Comparison Dashboard →")


# ══════════════════════════════════════════════
# TAB 4 — Comparison Dashboard
# ══════════════════════════════════════════════
with tab4:
    st.header("Comparison Dashboard")

    models_ready = (
        st.session_state.baseline  is not None and
        st.session_state.pruned    is not None and
        st.session_state.quantized is not None
    )

    if not models_ready:
        st.warning("Complete all three steps (Train → Prune → Quantize) to see the full comparison.")
    else:
        _, test_loader = get_dataloaders(batch_size)
        criterion      = nn.CrossEntropyLoss()

        models = {
            "Baseline"   : st.session_state.baseline,
            "Pruned"     : st.session_state.pruned,
            "Quantized"  : st.session_state.quantized,
        }

        rows = {}
        for name, m in models.items():
            _, acc    = evaluate(m, test_loader, criterion)
            total, nz = count_parameters(m)
            size_kb   = model_size_kb(m)
            infer_ms  = measure_inference_ms(m, test_loader)
            rows[name] = {
                "Accuracy (%)"  : round(acc, 2),
                "Non-zero Params": nz,
                "Model Size (KB)": round(size_kb, 2),
                "Inference (ms)" : round(infer_ms, 2),
            }

        import pandas as pd
        df = pd.DataFrame(rows).T
        st.subheader("Summary Table")
        st.dataframe(df.style.highlight_max(axis=0, color="#c8f7c5")
                              .highlight_min(axis=0, color="#fadadd"), use_container_width=True)

        # Bar charts
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        keys   = list(df.columns)

        for i, col in enumerate(keys):
            axes[i].bar(df.index, df[col], color=colors)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_ylabel(col.split("(")[-1].rstrip(")") if "(" in col else "")
            for tick in axes[i].get_xticklabels():
                tick.set_rotation(20)

        plt.suptitle("Model Optimization Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # Key takeaways
        base_size = rows["Baseline"]["Model Size (KB)"]
        prune_size = rows["Pruned"]["Model Size (KB)"]
        q_size     = rows["Quantized"]["Model Size (KB)"]

        st.subheader("🔑 Key Takeaways")
        col1, col2, col3 = st.columns(3)
        col1.info(
            f"**Pruning**\n\n"
            f"Size: {base_size:.1f} → {prune_size:.1f} KB\n\n"
            f"Reduction: **{(1-prune_size/base_size)*100:.1f}%**\n\n"
            f"Acc drop: **{rows['Baseline']['Accuracy (%)'] - rows['Pruned']['Accuracy (%)']:+.2f}%**"
        )
        col2.info(
            f"**Quantization**\n\n"
            f"Size: {base_size:.1f} → {q_size:.1f} KB\n\n"
            f"Reduction: **{(1-q_size/base_size)*100:.1f}%**\n\n"
            f"Acc drop: **{rows['Baseline']['Accuracy (%)'] - rows['Quantized']['Accuracy (%)']:+.2f}%**"
        )
        col3.success(
            "**Conclusion**\n\n"
            "Both techniques reduce model footprint significantly "
            "with minimal accuracy loss — ideal for **edge deployment**."
        )

st.markdown("---")
st.caption("Model Optimization: Pruning & Quantization")
