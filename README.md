<div align="center">

# 🏠 Deep Learning for Rooftop Detection, Segmentation & Material Classification

### Urban Building Footprint Extraction from High-Resolution Drone Imagery

<br>

| 🐍 Language | 🔥 Framework | 📡 Domain | 🗂️ Dataset | 🎓 Institution |
|:-----------:|:------------:|:---------:|:----------:|:--------------:|
| Python 3.9+ | PyTorch 2.1+ | Remote Sensing | Nacala-Roof-Material | SASTRA Deemed University |

<br>

**ICT400 Major Project — SASTRA Deemed to be University (2025–26)**

*Base Paper: Guthula et al. (2025), Science of Remote Sensing — [DOI: 10.1016/j.srs.2025.100306](https://doi.org/10.1016/j.srs.2025.100306)*

</div>

</div>



## 📋 Table of Contents

- [Why This Project Matters](#-why-this-project-matters)
- [What Makes This Hard](#-what-makes-this-hard)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Models at a Glance](#-models-at-a-glance)
- [Architecture Deep Dive](#-architecture-deep-dive)
- [The DOW Innovation](#-the-dow-innovation-solving-touching-buildings)
- [Training Pipeline](#-training-pipeline)
- [Loss Functions](#-loss-functions)
- [Class Imbalance Handling](#-class-imbalance-handling)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting](#-troubleshooting)
- [Base Paper & Extensions](#-base-paper--extensions)
- [Citations](#-citations)
- [Authors](#-authors)

---

## 🌍 Why This Project Matters

Malaria kills hundreds of thousands of people every year, with over 94% of cases occurring in sub-Saharan Africa. Research has established a direct link between **roof material and malaria risk**:

| Roof Type | Risk Level | Why |
|---|---|---|
| 🌾 Thatch | **High** | More mosquito entry points; cooler indoor climate favors mosquito survival |
| 🔩 Metal Sheet | **Lower** | Higher daytime temperatures; reduced mosquito survival |
| 🏗️ Concrete / Asbestos | **Lowest** | Well-sealed, climate-controlled interiors |

Manual field surveys to map roof types across thousands of buildings are expensive and slow. This project trains deep learning models to **automatically classify roof materials from drone images**, enabling health authorities to identify high-risk areas at scale and plan targeted interventions like insecticide-treated nets (ITNs) and indoor residual spraying (IRS).

---

## 🧩 What Makes This Hard

This problem has three distinct challenges that standard segmentation pipelines cannot handle out of the box:

### Challenge 1 — Touching Buildings
In dense informal settlements, buildings are so tightly packed that their roofs physically touch. A standard model sees them as one blob. You cannot count buildings or classify individual roofs if they are merged.

```
Ground truth:   [  Building A  ]  [gap]  [  Building B  ]
U-Net output:   [        Building AB (merged)           ]   ← WRONG
```

### Challenge 2 — Extreme Class Imbalance
```
Metal Sheet : 9,776 buildings  ████████████████████████  54.4%
Thatch      : 6,428 buildings  ████████████████          35.8%
No Roof     : 1,010 buildings  ██                         5.6%
Asbestos    :   566 buildings  █                          3.2%
Concrete    :   174 buildings  ▌                          1.0%  ← 50x fewer than Metal Sheet
```
A naive model ignores Concrete entirely and still achieves 90%+ pixel accuracy. Yet Concrete is critical for public health analysis.

### Challenge 3 — Two Conflicting Goals
- **High IoU** (pixel quality) → predict exact roof boundary → touching roofs must be merged
- **High AP50** (instance detection) → each building as a separate object → touching roofs must be split

These goals directly conflict. Our **Deep Ordinal Watershed (DOW)** approach resolves this using dual output heads.

---

## 📦 Dataset

### Nacala-Roof-Material Dataset

Captured by a DJI Phantom 4 Pro drone at 120m altitude over three informal settlements in Nacala, Mozambique (October–December 2021).

| Property | Details |
|---|---|
| Resolution | **4.4 cm/pixel** — extremely fine detail |
| Total buildings | **17,954** carefully annotated |
| Annotation | 3-step process: field survey → digitization → author verification (~160 person-hours) |
| Dataset link | https://mosquito-risk.github.io/Nacala |
| Interactive explorer | https://mosquito-risk.github.io/Nacala/demo |

### Class Distribution Across Splits

| Class | Train | Validation | Test | External |
|---|---|---|---|---|
| Metal Sheet | 3,696 | 1,093 | 978 | 661 |
| Thatch | 1,609 | — | — | 792 |
| No Roof | 626 | 149 | 149 | 86 |
| Asbestos | 394 | 77 | 95 | 0 |
| Concrete | 121 | 28 | 23 | 2 |
| **Total** | **6,093** | **1,609** | **1,282** | **792** |

### How the Splits Work

The dataset is split in a **spatially aware** way to prevent data leakage:

- **Train / Val / Test** — from two settlements, divided by a 225m grid (stratified by class so each split has a similar class distribution)
- **External (ext)** — an **entirely separate third settlement**, never seen during training; used purely to test geographic generalization

A building is assigned to a grid cell based on its centroid. If it straddles two cells from different splits, the overlapping pixels in the "other" split are masked out.

---

## 📁 Project Structure

```
.
├── review1_code.ipynb        # Primary experiments — UNet-DOW-Multi model
├── review2_code.ipynb        # DINOv2 model experiments
├── major_report__1_.docx     # Full project report (methodology, results, discussion)
└── README.md

# Auto-generated on first run:
├── checkpoints/
│   ├── {model}_best.pt                  # Best weights by validation IoU
│   ├── {model}_training_curves.png      # Loss / IoU / F1 / LR over epochs
│   ├── {model}_per_class_ap.png         # Per-class AP@50 bar chart
│   ├── {model}_confusion_matrix.png     # Normalized confusion matrix
│   └── {model}_metrics_heatmap.png      # All metrics in one view
│
├── results/{model}/
│   ├── fig*_rgb_gt_pred.png             # RGB / GT / Prediction comparison
│   ├── fig*_overlay.png                 # Prediction overlaid on RGB image
│   ├── fig*_error.png                   # TP(green) / FP(red) / FN(blue)
│   ├── fig*_dow.png                     # DOW area + interior maps
│   ├── fig*_instances.png               # Instance-colored segmentation
│   └── fig*_roof.png                    # 6-panel roof material figure
│
└── .nacala_cache/
    ├── train_offset0_precomputed.npz    # Preprocessed train arrays (binary, interior, weight, multi)
    └── val_offset0_precomputed.npz      # Preprocessed val arrays
```

> Both notebooks run the **same master pipeline** — only the `MODELS_TO_RUN` variable differs.

---

## 🗂️ Models at a Glance

This project implements and benchmarks **17 model variants** across 5 architecture families.

### Quick Selection Guide

| Use Case | Recommended Model |
|---|---|
| Best overall performance | `mega_ensemble` |
| Best single model (balanced) | `unet_dow_multi` or `dinov2_dow_multi` |
| Best binary segmentation only | `unet` |
| Fastest training / limited GPU | `unet` or `yolov8_seg` |
| Studying attention mechanisms | `yolov11_seg` (AIFI) or `yolov12_seg` (Area Attention) |
| Studying DOW concept | Compare `unet` vs `unet_dow` side-by-side |

### All 17 Variants

| Family | Model | Key Feature | Output |
|---|---|---|---|
| **U-Net** | `unet` | ResNet34 + skip connections | Binary |
| | `unet_dow` | + DOW interior head | Binary + Interior |
| | `unet_multi` | + 6-class roof head (shared decoder) | Binary + Roof |
| | `unet_dow_multi` ⭐ | Dual U-Net++ decoders | Binary + Interior + Roof |
| **DINOv2** | `dinov2` | Frozen ViT + conv decoder | Binary |
| | `dinov2_multi` | + roof classification head | Binary + Roof |
| | `dinov2_dow` | + DOW interior head | Binary + Interior |
| | `dinov2_dow_multi` ⭐ | Dual 5-stage decoders | Binary + Interior + Roof |
| **YOLO** | `yolov8_seg` / `yolov8_seg_multi` | FPN neck, multi-scale features | Binary / + Roof |
| | `yolov11_seg` / `yolov11_seg_multi` | + C3k2 blocks + AIFI attention | Binary / + Roof |
| | `yolov12_seg` / `yolov12_seg_multi` | + R-ELAN + Area Attention | Binary / + Roof |
| **Novel** | `deeplab_multi` | ResNet50 + ASPP dilated convolutions | Binary + Interior + Roof |
| | `pspnet_multi` | ResNet34 + Pyramid Pooling Module | Binary + Interior + Roof |
| | `segnet_multi` | Index-based unpooling + SE attention | Binary + Interior + Roof |
| **Ensemble** | `ensemble_unet_dino` | Learnable per-channel fusion (U-Net + DINOv2) | Binary + Interior + Roof |
| | `mega_ensemble` ⭐ | Adaptive spatial gating (all 8 models) | Binary + Interior + Roof |

---

## 🏗️ Architecture Deep Dive

### U-Net Family

All U-Net variants share a **ResNet34 encoder** pretrained on ImageNet, which produces feature maps at 5 scales (strides 2, 4, 8, 16, 32). The decoder upsamples via nearest-neighbor and concatenates encoder skip connections at each scale.

**`unet`** — Baseline with two output heads:
```
ResNet34 → UNet Decoder [B, 16, H, W]
    ├── head_binary   (3×3 Conv → BN → ReLU → 1×1 Conv) → [B, 1, H, W]
    └── head_interior (larger 64-filter head)             → [B, 1, H, W]
```

**`unet_dow_multi`** ⭐ — Two completely separate **U-Net++ decoders** from one shared encoder:
```
ResNet34 Encoder (shared)
    ├── UNet++ Spatial Decoder  → head_binary + head_interior
    └── UNet++ Semantic Decoder → head_roof (6 classes)
```
U-Net++ uses **nested dense skip connections** — each decoder node receives inputs from all previous nodes at that resolution level, preserving finer boundary detail compared to standard U-Net.

---

### DINOv2 Family

DINOv2 is a **Vision Transformer (ViT)** trained with self-supervised learning on 142M images. The encoder is completely **frozen** — only the decoder is trained. This gives rich semantic features without needing massive labeled datasets.

```python
# DINOv2 processes 448×448 images as 32×32 patches of dim 768
tokens   = dinov2(x).last_hidden_state[:, 1:, :]      # [B, 1024, 768]
features = tokens.reshape(B, 32, 32, 768).permute(0,3,1,2)  # [B, 768, 32, 32]
```

**`dinov2_dow_multi`** ⭐ — Two separate **5-stage** decoders (one extra stage vs. base DINOv2):
```
DINOv2 Features [B, 768, 32, 32]
    ├── Spatial Decoder  (768→512→256→128→64→32) → binary + interior
    └── Semantic Decoder (768→512→256→128→64→32) → roof classes
```
GELU activations are used instead of ReLU throughout DINOv2 decoders — smoother gradients suit transformer-derived features better.

---

### YOLO Family

All YOLO variants use a **ResNet34 + Feature Pyramid Network (FPN)** backbone implemented in pure PyTorch (no Ultralytics library required).

**FPN multi-scale fusion:**
```
ResNet34 features:  f1[B,64,H/4,W/4]  f2[B,128,H/8,W/8]  f3[B,256,H/16,W/16]  f4[B,512,H/32,W/32]
                              ↓ lateral projections → all 128ch
Top-down path:  p4 → p3 = p3 + upsample(p4) → p2 = p2 + upsample(p3) → p1 ...
Final: concat all 4 levels at 1/4 resolution → 512ch → 128ch → 64ch → full resolution
```

**YOLOv11 additions:**
- **C3k2 (Cross-Stage Partial with dual kernels)** — two depthwise-separable conv branches merged; more efficient than standard convolutions
- **AIFI (Area-Invariant Feature Interaction)** — multi-head self-attention on the deepest FPN feature map (downsampled to 16×16 for efficiency); captures global settlement context

**YOLOv12 additions:**
- **Area Attention (A²)** — self-attention computed within local 8×8 windows; O(A × n²) vs O(HW)² for global attention; perfect for dense building clusters
- **R-ELAN (Residual Efficient Layer Aggregation)** — dense residual connections across depthwise-separable conv branches; faster gradient flow, especially helpful for rare classes

---

### Novel Architectures

**`deeplab_multi`** — ResNet50 with dilated convolutions (stride-8 output) + ASPP:
```
ASPP: 5 parallel branches (1×1 conv, dilated conv rates 6/12/18, global avg pool)
→ Concatenated → 256ch → low-level skip (projected to 48ch) → full resolution
```
*Note: Good IoU (0.8346) but lower AP50 (0.5998) — ASPP captures regions well but tends to merge touching buildings.*

**`pspnet_multi`** — Pyramid Pooling Module for scene-level understanding:
```
PPM: 4 pooling scales (1×1, 2×2, 3×3, 6×6) → upsample + concat → 1024ch → 256ch
Deep supervision auxiliary head at stride-8 for faster gradient propagation
```

**`segnet_multi`** — Index-based unpooling with SE (Squeeze-and-Excitation) channel attention:
```
Encoder: Conv-BN-ReLU → MaxPool (saves indices) × 5 stages
Decoder: MaxUnpool(saved indices) → Conv → SEBlock × 5 stages
SEBlock: GlobalAvgPool → FC → ReLU → FC → Sigmoid → channel-wise scale
```

---

### Ensemble Models

**`ensemble_unet_dino`** — Learnable per-channel branch weights:
```python
w = torch.softmax(fusion_logits, dim=0)   # [2, 8] — 2 branches × 8 output channels
output = w[0] * unet_output + w[1] * dino_output
# Binary channels may trust U-Net more; roof class channels may trust DINOv2 more
```

**`mega_ensemble`** ⭐ — Adaptive spatial gating network across 8 models:
```
gate_net: 3ch → 32ch (stride=4) → 64ch (stride=4) → (n_branches × 8)ch
        → bilinear upsample → softmax over branches → [B, 8_models, 8_channels, H, W]

Final: 0.3 × global_weights + 0.7 × spatial_weights
```
Different image regions can be handled by different models — dense urban zones may weight U-Net more; areas with rare roof types may weight DINOv2 more.

---

## 💡 The DOW Innovation — Solving Touching Buildings

### The Problem

```
Ground truth:    [ Building A ]  [gap]  [ Building B ]
Standard U-Net:  [      Building AB (merged)         ]  → counts 1, misclassifies both
```

### The Solution

The Deep Ordinal Watershed (DOW) method predicts **two masks simultaneously**:

```
Level 1 (Area):     [   building A   ]  [gap]  [   building B   ]
Level 2 (Interior): [  interior A  ]            [  interior B  ]
```

The interior mask is simply the building eroded by **10 pixels**. Even when two buildings touch, their interiors remain separated (as long as buildings are >10px wide).

### How the Watershed Works

```python
# 1. Distance transform of binary mask → treats buildings as "terrain"
dist    = ndi.distance_transform_edt(binary_mask)

# 2. Interior mask gives one seed per building
markers = ndi.label(interior_mask)[0]

# 3. Watershed floods outward from each interior seed
#    Where two floods meet = building boundary
instances = watershed(-dist, markers, mask=binary_mask)
```

### Why This Resolves the Conflict

| Head | Objective | Metric |
|---|---|---|
| Head 1 (Area) | Maximize overlap with full roof → predict touching roofs as merged | IoU ↑ |
| Head 2 (Interior) | Predict only the core interior → always separated | AP50 ↑ |

The model achieves **high IoU and high AP50 simultaneously**, which a single-head model cannot do.

### Building the Training Target

```python
dist     = ndi.distance_transform_edt(binary_mask)
interior = (dist > DOW_NPIX).astype(np.uint8)   # 1 if > 10px from any edge
# Small buildings (<10px radius) → empty interior → fallback to binary mask
```

---

## ⚙️ Training Pipeline

```
Raw drone orthomosaic
        ↓
Patch extraction  (512×512 for U-Net/DeepLab, 640×640 for YOLO, 448×448 for DINOv2)
        ↓
Cache-based preprocessing (built once, reused forever)
  ├── Binary mask from int_mask/ .tif files
  ├── DOW interior mask  (Euclidean distance transform → threshold at n_pix=10)
  ├── Multi-class mask   (rasterized from YOLO .txt polygon labels)
  ├── Border relabeling  (7px gap enforced between touching buildings)
  └── Optional pixel-wise weight maps  (very slow — disabled by default)
        ↓
Training loop  (AdamW, lr=3e-4, batch=16, 25 epochs)
  ├── Data augmentation: H/V flips, rotation, brightness/contrast jitter
  ├── AMP (FP16) for ~1.5–2× speedup on Ampere+ GPUs
  ├── torch.compile for ~20–30% additional speedup (PyTorch 2.0+)
  ├── Gradient clipping at max_norm=5.0
  ├── Validation every 5 epochs (no TTA for speed)
  └── Best checkpoint saved by validation IoU
        ↓
Final evaluation  (on best checkpoint)
  ├── Threshold search: single forward pass, sweep [0.30 … 0.60], pick best IoU
  ├── Test-Time Augmentation: 3 passes (original + H-flip + V-flip) → averaged logits
  ├── Morphological post-processing: remove connected components < 16 pixels
  └── DOW watershed: interior seeds → separated building instances
        ↓
Metrics: IoU · mIoU3 · mIoU5 · AP50 · mAP50 · AP50-95 · Boundary F1 · TPs
```

### LR Schedule

**Default (cosine annealing with warmup):**
```
Epochs 1–10 : lr = (epoch/10) × base_lr        ← linear warmup
Epochs 11–N : lr = 0.5 × base_lr × (1 + cos(π × progress))  ← cosine decay
```
Warmup prevents large early updates from overwhelming rare class gradients before the model has seen enough examples.

**Paper-faithful (`USE_PAPER_STEPLR=True`):** StepLR — decay by 0.1 every 50 epochs. Only useful for 300-epoch runs.

---

## 📐 Loss Functions

### Total Loss

```
L_total = L_binary + L_interior + L_multi
```

### Binary Segmentation Loss

```
L_binary = BCE_smooth + 0.5 × Dice + 0.5 × Tversky
```

| Component | Formula | Why |
|---|---|---|
| **BCE** (label-smoothed) | targets: 1→0.9, 0→0.05 | Prevents overconfident background predictions |
| **Dice** | `1 - (2×TP) / (2×TP + FP + FN)` | Focuses on overlap; robust when buildings are small |
| **Tversky** | `1 - TP / (TP + 0.3×FP + 0.7×FN)` | Penalizes missed buildings (FN) more than false alarms |

Pixel weight map (when enabled):
```
w(x) = w₀ × exp( −(d₁(x) + d₂(x))² / (2σ²) )   [w₀=10, σ=5]
```
Where d₁, d₂ are distances to the two nearest building borders. Background pixels between touching buildings get the highest weight, forcing the model to learn the gap.

### DOW Interior Loss

```
L_interior = 2.0 × BCE(interior_pred, interior_target)
```
The 2× weight compensates for slower convergence — the interior task is harder and needs a stronger gradient signal.

### Multi-Class Roof Loss

```
L_multi = 3.0 × FocalCE(roof_logits, roof_target, class_weights)
```

**Focal Cross-Entropy:**
```
FocalCE = (1 − p_correct)^γ × CE,    γ = 2.0
```
When the model is confident and correct, `(1-p)^γ ≈ 0` — easy examples barely contribute. Hard examples (wrong or uncertain) dominate the loss, forcing attention to rare classes like Concrete.

`MULTI_LOSS_MASK_ONLY=True` — CE computed only inside building pixels. Background excluded because (a) it has no roof class label and (b) it would overwhelm the foreground signal.

`MULTI_LOSS_WEIGHT=3.0` — Without this amplification, the multi-class signal is drowned out by the binary loss.

---

## ⚖️ Class Imbalance Handling

Three complementary strategies are combined:

### 1. Median-Frequency Class Weights

Computed automatically by scanning the full training set once:
```python
freq_c    = (pixels of class c) / (total pixels)
weight_c  = median(all_freqs) / freq_c
```

Approximate resulting weights for this dataset:
```
Background  →  0.05   (suppressed — handled by binary head)
Metal Sheet →  1.00   (baseline)
Thatch      →  1.50
Asbestos    →  ~8.0
Concrete    →  ~25.0  ← heavily boosted
No Roof     →  ~10.0
```
Weights are capped at 50× the minimum to prevent gradient explosion.

### 2. Focal Loss (γ = 2.0)
Even with class weights, easy examples still dominate. Focal loss down-weights confident correct predictions, forcing the model to concentrate on Concrete and Asbestos examples where it is still uncertain.

### 3. Mask-Only CE
Computing CE loss only on foreground pixels eliminates the vast background-to-foreground imbalance. Without this, the model learns to predict "background" everywhere and still achieves low loss.

---

## 📊 Evaluation Metrics

| Metric | What It Measures | Range |
|---|---|---|
| **IoU** | Pixel-level overlap: `|Pred ∩ GT| / |Pred ∪ GT|` | 0 → 1 |
| **mIoU3** | Mean IoU over Metal Sheet, Thatch, Asbestos | 0 → 1 |
| **mIoU5** | Mean IoU over all 5 roof classes | 0 → 1 |
| **AP50** | Instance detection accuracy at IoU ≥ 0.5 | 0 → 1 |
| **mAP50** | Mean AP50 across roof material classes (class must also be correct) | 0 → 1 |
| **AP50-95** | COCO-style: mean AP over IoU thresholds 0.50 → 0.95 (step 0.05) | 0 → 1 |
| **Boundary F1** | Precision/recall of predicted edges within 2px of true edges | 0 → 1 |
| **TPs** | Count of buildings correctly detected (IoU > 0.5) — out of 2,527 in test set | integer |

**AP50 computation:**
1. Build IoU matrix [Predictions × GT buildings]
2. Greedily match predictions to GT (best IoU first)
3. Unmatched predictions → FP; unmatched GT → FN
4. AP50 = area under Precision-Recall curve

---

## 📈 Results

### Model Comparison (25 training epochs, validation set)

| Model | Pixel IoU | Object AP50 | Notes |
|---|---|---|---|
| `unet` | 0.8586 | 0.8393 | Best binary-only IoU |
| `unet_dow` | 0.8563 | 0.8384 | DOW slightly trades IoU for better instance separation |
| `unet_multi` | 0.8173 | 0.7461 | Shared decoder — gradient competition hurts |
| `unet_dow_multi` | 0.8368 | 0.7965 | Separate decoders recover most loss |
| `dinov2` | 0.8416 | 0.8295 | Strong semantics from ViT features |
| `dinov2_dow` | 0.8413 | 0.8277 | — |
| `dinov2_multi` | 0.8279 | 0.8130 | — |
| `dinov2_dow_multi` | 0.8431 | 0.8289 | Best DINOv2 variant |
| `yolov8_seg` | 0.8531 | 0.8431 | Fast and accurate |
| `yolov8_seg_multi` | 0.8316 | 0.7826 | Multi-task cost on YOLO |
| `yolov12_seg` | 0.8556 | 0.8319 | Best single YOLO |
| `yolov12_seg_multi` | 0.8310 | 0.7130 | — |
| `deeplab_multi` | 0.8346 | 0.5998 | Good IoU, poor instance separation |
| `pspnet_multi` | 0.8266 | 0.7582 | Balanced performance |
| `segnet_multi` | 0.6717 | 0.4129 | Needs 100+ epochs to converge |
| `ensemble_unet_dino` | 0.8671 | 0.8405 | Best ensemble |
| **`mega_ensemble`** | **0.8671** | **0.8405** | **Best overall** |

### Paper Baseline (300 epochs, test set — Guthula et al. 2025)

| Model | IoU | AP50 |
|---|---|---|
| U-Net | 0.895 | 0.910 |
| U-NetDOW | 0.895 | 0.935 |
| DINOv2 | 0.880 | 0.881 |
| DINOv2DOW | 0.881 | 0.931 |
| YOLOv8 | 0.866 | 0.941 |

> Our 25-epoch results are competitive baselines. Paper results use 300 epochs, pixel weight maps, and 5-trial averaging.

---

## 🛠️ Installation

### Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| PyTorch | 1.13 | 2.1+ |
| GPU VRAM | 8 GB | 16 GB |
| Disk space | 7 GB | 20 GB |
| RAM | 16 GB | 32 GB |

### Step-by-Step

```bash
# 1. Create a virtual environment
python -m venv nacala_env
source nacala_env/bin/activate        # Linux / Mac
# nacala_env\Scripts\activate.bat    # Windows

# 2. Install PyTorch (adjust cu118 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only: pip install torch torchvision

# 3. Install segmentation and vision libraries
pip install segmentation-models-pytorch
pip install transformers huggingface-hub     # for DINOv2

# 4. Install geospatial libraries
pip install rasterio geopandas rasterstats shapely

# 5. Install scientific stack
pip install numpy pandas matplotlib seaborn scikit-image scipy tqdm

# 6. Install CV and metrics
pip install opencv-python Pillow torchmetrics

# 7. Optional (for some YOLO utilities only)
pip install ultralytics
```

### Verify

```python
import torch, rasterio, segmentation_models_pytorch as smp
from transformers import Dinov2Model

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## 🚀 How to Run

### 1. Set dataset paths

In the notebook, update:
```python
TRAIN_ROOT = "/path/to/nacala/train"
VALID_ROOT  = "/path/to/nacala/valid"
```

### 2. Choose your model

```python
MODELS_TO_RUN = ["unet_dow_multi"]   # pick from the list below
```

**All available model names:**
```
unet              unet_multi        unet_dow          unet_dow_multi
dinov2            dinov2_multi      dinov2_dow        dinov2_dow_multi
yolov8_seg        yolov8_seg_multi
yolov11_seg       yolov11_seg_multi
yolov12_seg       yolov12_seg_multi
deeplab_multi     pspnet_multi      segnet_multi
ensemble_unet_dino                  mega_ensemble
```

### 3. Adjust for your GPU

```python
# 8 GB VRAM
PARAMS["batch_size"] = 8
TRAIN_IMG_SIZE = 256

# 16 GB VRAM (default)
PARAMS["batch_size"] = 16
TRAIN_IMG_SIZE = 384

# 40+ GB VRAM
PARAMS["batch_size"] = 32
TRAIN_IMG_SIZE = 512
```

### 4. Run

Open `review1_code.ipynb` or `review2_code.ipynb` in Jupyter and **run all cells top to bottom**.

On first run, you will see:
1. Dataset diagnostic (confirms masks and labels are loading correctly)
2. Cache build progress (~5 min for the full dataset)
3. Class weight computation scan
4. Training epochs with per-epoch loss, IoU, and timing
5. Final evaluation with the full metrics table
6. Saved visualizations in `results/` and `checkpoints/`

> **Cache note:** The pipeline saves preprocessed training arrays to `.nacala_cache/` on the first run. Subsequent runs skip preprocessing entirely and start training immediately.
>
> If you change `DOW_NPIX`, `BORDER_GAP_PX`, or `MASK_CLASS_OFFSET`, **always delete the cache first:**
> ```bash
> rm -rf .nacala_cache/
> ```

---

## 🔧 Configuration Reference

All settings are global variables at the top of each notebook.

```python
# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_ROOT        = "/path/to/train"
VALID_ROOT        = "/path/to/valid"
MASK_CLASS_OFFSET = 0          # 0 = masks already 1-indexed (Nacala default)
                               # ALWAYS delete cache when changing this!

# ── Training ───────────────────────────────────────────────────────────────
PARAMS = {
    "lr":           3e-4,      # AdamW learning rate
    "batch_size":   16,        # Reduce to 4–8 if low VRAM
    "epochs":       25,        # Use 150–300 for paper-faithful results
    "weight_decay": 1e-4,
}
TRAIN_IMG_SIZE = 384           # Training resolution (256=fastest, 512=best quality)
EVAL_IMG_SIZE  = 512           # Evaluation resolution (keep at 512)

# ── Model selection ────────────────────────────────────────────────────────
MODELS_TO_RUN = ["unet_dow_multi"]

# ── DOW settings ───────────────────────────────────────────────────────────
DOW_NPIX           = 10        # Interior erosion margin in pixels (paper default)
USE_BORDER_RELABEL = True      # Enforce 7px gap between touching buildings
BORDER_GAP_PX      = 7
USE_WEIGHT_MAP     = False     # Pixel weight maps — very slow (~2h cache build)
WEIGHT_MAP_W0      = 10.0
WEIGHT_MAP_SIGMA   = 5.0

# ── Loss ───────────────────────────────────────────────────────────────────
AUTO_COMPUTE_CLASS_WEIGHTS  = True    # Scan train set to compute class weights
MULTI_LOSS_WEIGHT           = 3.0    # Amplifier for multi-class CE loss
MULTI_LOSS_MASK_ONLY        = True   # CE only on foreground pixels
USE_FOCAL_IN_LOSS           = False  # Extra focal term for binary loss
FOCAL_GAMMA                 = 2.0
FOCAL_ALPHA                 = 0.75

# ── Inference ──────────────────────────────────────────────────────────────
USE_TTA              = True           # 3-augmentation test-time averaging
USE_THRESHOLD_SEARCH = True           # Search optimal binarization threshold
THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
USE_MORPH_POST       = True           # Remove tiny segments
MORPH_MIN_SIZE       = 16             # Minimum segment size in pixels

# ── LR schedule ────────────────────────────────────────────────────────────
USE_PAPER_STEPLR = False   # True = StepLR(step=50, γ=0.1) for 300-epoch runs
                           # False = CosineAnnealing + 10-epoch warmup (default)

# ── Hardware ───────────────────────────────────────────────────────────────
_MAX_WORKERS = 32          # DataLoader workers (too high causes FD exhaustion)
```

---

## 🩺 Troubleshooting

**"All masks appear to be all-zeros"**
```python
# Check your mask files directly:
import rasterio, numpy as np
with rasterio.open("path/to/int_mask/00000001.tif") as src:
    print(np.unique(src.read(1)))    # should print [0, 1]
```

**"No roof classes found in YOLO labels"**
```bash
# Check filenames match between images/ and labels/
ls labels/ | head -5
ls images/ | head -5
# e.g., 00000001.tif  ↔  00000001.txt
```

**"CUDA out of memory"**
```python
PARAMS["batch_size"] = 4     # or 8
TRAIN_IMG_SIZE = 256          # reduce resolution
```

**DINOv2 download fails**
```bash
# Pre-download the model (~330 MB)
python -c "from transformers import Dinov2Model; Dinov2Model.from_pretrained('facebook/dinov2-base')"
# Or set a custom cache directory:
export HF_HOME="/path/to/fast/storage"
```

**"Found stale caches with different offsets"**
```bash
rm -rf .nacala_cache/
```

**SegNet has very poor results at 25 epochs**
This is expected. SegNet's architecture was designed for lower-resolution inputs and needs 100+ epochs to converge on drone imagery. Use a different model or increase `PARAMS["epochs"]`.

---

## 📄 Base Paper & Extensions

### Base Paper

> **Guthula, V.B., Oehmcke, S., Chilaule, R., Zhang, H., Lang, N., Kariryaa, A., Mottelson, J., & Igel, C. (2025)**
> *"Drone imagery for roof detection, classification, and segmentation to support mosquito-borne disease risk assessment: The Nacala-Roof-Material dataset"*
> Science of Remote Sensing, Vol. 12, p. 100306.
> [DOI: 10.1016/j.srs.2025.100306](https://doi.org/10.1016/j.srs.2025.100306)

### What We Replicated from the Paper

- U-Net with ResNet34 encoder (ImageNet pretrained)
- DINOv2-based segmentation decoder (frozen encoder)
- YOLOv8 instance segmentation
- Deep Ordinal Watershed (DOW) for instance separation
- AP50, mIoU3/mIoU5, AP50-95, and TPs evaluation protocol
- Border relabeling and pixel-wise weight maps during training

### What We Added Beyond the Paper

| Extension | Description |
|---|---|
| YOLOv11 + YOLOv12 | Novel architectures with C3k2/AIFI and R-ELAN/Area Attention |
| DeepLabV3+ | ASPP dilated convolutions for multi-scale context |
| PSPNet | Pyramid Pooling for scene-level understanding |
| SegNet | Index-based unpooling with SE attention |
| Mega-Ensemble | Adaptive spatial gating network across 8 model branches |
| Boundary F1 | Precision/recall of predicted building edges (2px tolerance) |
| TTA | 3-augmentation test-time averaging |
| Morphological post-processing | Removal of sub-16px spurious detections |
| Auto class weight computation | Median-frequency balancing from training set scan |

---

## 📚 Citations

```bibtex
@article{guthula2025nacala,
  title     = {Drone imagery for roof detection, classification, and segmentation
               to support mosquito-borne disease risk assessment:
               The Nacala-Roof-Material dataset},
  author    = {Guthula, Venkanna Babu and Oehmcke, Stefan and Chilaule, Remigio
               and Zhang, Hui and Lang, Nico and Kariryaa, Ankit
               and Mottelson, Johan and Igel, Christian},
  journal   = {Science of Remote Sensing},
  volume    = {12},
  pages     = {100306},
  year      = {2025},
  doi       = {10.1016/j.srs.2025.100306}
}

@inproceedings{ronneberger2015unet,
  title     = {{U-Net}: Convolutional networks for biomedical image segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {MICCAI},
  year      = {2015}
}

@article{oquab2024dinov2,
  title   = {{DINOv2}: Learning robust visual features without supervision},
  author  = {Oquab, Mathilde and others},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}
```

---

## 👥 Authors

**ICT400 Major Project — SASTRA Deemed to be University, 2025–26**

| Name | Roll Number |
|---|---|
| Bilavampalli Dinesh Reddy | 126014011 |
| Daliparthy Ravi Krishna Charan | 126014014 |

*Built upon the Nacala-Roof-Material benchmark by Guthula et al., University of Copenhagen.*
*Dataset exploration: https://mosquito-risk.github.io/Nacala/demo*
*Funding (original paper): Novo Nordisk Foundation (NNF21OC0069116), Pioneer Centre for AI (DNRF P1)*
