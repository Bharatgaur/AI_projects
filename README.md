# Model Optimization: Pruning & Quantization

---

## Overview

This practical demonstrates two industry-standard techniques to **shrink and speed up** deep learning models:

| Technique | Method | Goal |
|-----------|--------|------|
| **Pruning** | L1-Unstructured (PyTorch) | Zero out small weights → reduce active parameters |
| **Quantization** | Dynamic Post-Training (INT8) | Reduce weight precision → smaller size, faster CPU inference |

**Dataset:** MNIST (auto-downloaded via torchvision)  
**Model:** Custom lightweight CNN (2 Conv + 2 FC layers)

---

## Project Structure

```
model-optimization/
│
├── app.py              ← Streamlit interactive dashboard (recommended)
├── main.py             ← Terminal-only script (quick demo)
├── notebook.ipynb      ← Jupyter notebook (step-by-step)
│
├── environment.yml     ← Conda environment (recommended)
├── requirements.txt    ← pip install alternative
│
├── data/               ← Auto-created; MNIST downloaded here
├── baseline_model.pt   ← Saved after training
├── pruned_model.pt     ← Saved after pruning
├── quantized_model.pt  ← Saved after quantization
│
└── README.md
```

---

## Setup Instructions

### Step 1 — Create Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate model-optimization
```

### Step 2 — OR Install via pip

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A — Streamlit App (Interactive UI)

```bash
streamlit run app.py
```
Opens at **http://localhost:8501** in your browser.

### Option B — Terminal Script

```bash
python main.py
```
Prints a comparison table directly in the terminal.

### Option C — Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```
Run cells one by one for a guided, step-by-step walkthrough.

---

## Expected Output

### Terminal (main.py)

```
======================================================================
 Pruning & Quantization
======================================================================

[1/3] Training baseline CNN …
  Epoch 1/3  train_loss=0.1821  val_acc=96.72%
  Epoch 2/3  train_loss=0.0641  val_acc=98.15%
  Epoch 3/3  train_loss=0.0476  val_acc=98.54%

[2/3] Applying L1-Unstructured Pruning (50%) …
[3/3] Applying Dynamic Post-Training Quantization (FP32 → INT8) …

======================================================================
Model           Accuracy    Size (KB)   Inference (ms)    Sparsity
----------------------------------------------------------------------
Baseline           98.54%       685.2            12.34        0.0%
Pruned             98.21%       685.2             9.87       50.0%
Quantized          98.47%       185.4             8.12        0.0%
======================================================================
```

### Streamlit Dashboard

| Tab | What you see |
|-----|-------------|
|  **Train Baseline** | Training curves (accuracy & loss per epoch), model stats |
|  **Pruning** | Before/after metrics, weight distribution histogram |
|  **Quantization** | Size & speed comparison vs baseline |
|  **Comparison Dashboard** | Full table + bar charts for all 3 models |

---

##  Key Concepts

### Pruning
```python
import torch.nn.utils.prune as prune

# Remove 50% of weights with lowest absolute value
prune.l1_unstructured(model.fc1, name='weight', amount=0.50)
prune.remove(model.fc1, 'weight')  # Make permanent
```

### Dynamic Quantization
```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # Quantize Linear layers
    dtype=torch.qint8    # FP32 → INT8
)
```

---

##  Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch torchvision` |
| `ModuleNotFoundError: streamlit` | Streamlit not installed | `pip install streamlit` |
| Download hangs | Firewall / proxy | Pre-download MNIST or use VPN |
| `RuntimeError: CUDA not available` | No GPU | Already uses CPU — no action needed |
| Quantized model won't save | Dynamic quant uses special format | Normal; still loads with `torch.load()` |
| Low accuracy (<90%) | Too few epochs | Increase epochs in sidebar to 5–10 |

---

##  Theory Summary

```
FP32 (32-bit float) → INT8 (8-bit integer)
- 4× smaller per value
- Faster CPU math (SIMD integer ops)
- <1% accuracy drop typical

Pruning (50% sparsity):
- Half the weights are zero
- Sparse matrix libraries can skip zeros
- No architecture change needed
```

---

##  Tools Used

- **PyTorch 2.2** — Model training, pruning, quantization
- **torchvision** — MNIST dataset
- **Streamlit 1.33** — Interactive web dashboard
- **Matplotlib** — Plots & charts
- **Pandas** — Comparison table

---