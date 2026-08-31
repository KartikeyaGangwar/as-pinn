# Adaptive $N$-Subspace Physics-Informed Neural Networks (AS-PINN)

Official PyTorch implementation of **Adaptive $N$-Subspace Physics-Informed Neural Networks (AS-PINN)**: a fully autonomous, data-driven domain decomposition and parameter Adaptive Mesh Refinement (AMR) framework for solving stiff, multi-scale, and high-dimensional partial differential equations (PDEs).

---

## 🌟 Key Innovations

1. **Vectorized Intra-Subspace Gradient Conflict Profiling:**
   Computes sample-wise Gram alignment matrices $\mathcal{G}_{ij} = \cos \angle(\mathbf{g}_i, \mathbf{g}_j)$ using batched forward-mode and reverse-mode automatic differentiation (`torch.func.vmap`), detecting localized destructive gradient interference in real time.

2. **Autonomous Zero-Disruption Parameter Cleavage:**
   Autonomously allocates and centers new localized subspace neural networks at spatial centers of severe gradient conflict, with mathematical guarantee of exact solution invariance ($\|u^{(N+1)} - u^{(N)}\| = 0$) upon fission.

3. **Dimension-Normalized Anisotropic Bandwidth Vector:**
   Eliminates coordinate aspect-ratio bias in multi-dimensional and space-time domains using coordinate-scaled bandwidth vectors $\boldsymbol{\sigma} = \gamma \cdot (\mathbf{b}_{\max} - \mathbf{b}_{\min})$ and scale-invariant non-redundancy distance metrics.

4. **Two-Stage Discover-and-Deploy Workflow:**
   * **Stage 1 (Deep AMR Discovery):** Lightweight probe autonomously uncovers minimal parsimonious subspace count $N^*$ and optimal centroids $\{\mathbf{c}_k\}$.
   * **Stage 2 (Production Deployment):** Clean $N^*$-subspace model is trained with synchronized AdamW and polished with Combined Global L-BFGS (Strong Wolfe).

---

## 📊 Comprehensive 9-Benchmark Quantitative Matrix

| Benchmark System | Domain & PDE Type | Standard PINN | PCGrad | CAGrad | **AS-PINN (Ours)** | Discovered $N^*$ | Relative $L^2$ Error | Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Burgers 1D** | 1D Shock ($\nu = 0.01/\pi$) | $2.70 \times 10^{-1}$ | $6.10 \times 10^{0}$ | $1.03 \times 10^{1}$ | $\mathbf{5.37 \times 10^{-2}}$ | $N^*=7$ | $60.8\%$ | **$5.0\times$ lower loss** |
| **Convection 2D** | Advection ($\boldsymbol{\beta}=(10, 10)$) | $1.71 \times 10^{0}$ | $4.58 \times 10^{0}$ | $9.83 \times 10^{0}$ | $\mathbf{1.69 \times 10^{-2}}$ | $N^*=22$ | **$3.24\%$** | **$100.6\times$ lower loss** ($18.2\times$ better $L^2$) |
| **Quantum NLS** | 2D BEC Soliton | $1.74 \times 10^{-2}$ | $2.59 \times 10^{-1}$ | $4.07 \times 10^{1}$ | $\mathbf{1.19 \times 10^{-2}}$ | $N^*=21$ | $42.0\%$ | **$1.5\times$ lower loss** |
| **Helmholtz 2D** | High-Freq ($k=4\pi$) | $1.52 \times 10^{3}$ | $3.09 \times 10^{3}$ | $3.24 \times 10^{3}$ | $\mathbf{2.09 \times 10^{0}}$ | $N^*=21$ | $22.1\%$ | **$725.6\times$ lower loss** |
| **Diffusion 4D** | 4D Space + 1D Time | $7.52 \times 10^{-1}$ | $1.46 \times 10^{0}$ | $1.13 \times 10^{1}$ | $\mathbf{3.33 \times 10^{-1}}$ | $N^*=24$ | $33.5\%$ | **$2.3\times$ lower loss** |
| **2D Bratu** | Thermal Runaway ($e^u$) | $1.18 \times 10^{-4}$ | $2.48 \times 10^{-3}$ | $2.81 \times 10^{1}$ | $\mathbf{1.56 \times 10^{-5}}$ | $N^*=17$ | **$0.0148\%$** | **$7.5\times$ lower loss** |
| **Kovasznay** | Navier-Stokes Wake ($\mathrm{Re}=40$) | $2.97 \times 10^{-3}$ | $2.73 \times 10^{0}$ | $2.05 \times 10^{1}$ | $\mathbf{2.21 \times 10^{-4}}$ | $N^*=7$ | **$0.188\%$** | **$13.4\times$ lower loss** |
| **Cavity NS** | Incompressible ($\mathrm{Re}=100$) | $2.81 \times 10^{-1}$ | $2.56 \times 10^{-1}$ | $4.09 \times 10^{0}$ | $\mathbf{1.11 \times 10^{-1}}$ | $N^*=22$ | **$9.39\%$** | **$2.5\times$ lower loss** ($8.0\times$ better $L^2$) |
| **Klein-Gordon** | Relativistic Wave ($u^3$) | $1.69 \times 10^{0}$ | $2.67 \times 10^{1}$ | $4.02 \times 10^{3}$ | $\mathbf{1.86 \times 10^{-1}}$ | $N^*=21$ | **$1.94\%$** | **$9.1\times$ lower loss** |

---

## 🛠️ Repository Structure

```
Adaptive_N_Subspace_PINN/
├── models/
│   ├── as_pinn.py              # AdaptiveSubspacePINN core architecture & Voronoi PoU
│   └── conflict_monitor.py     # Fast vectorized Gram conflict analyzer & centroid locator
├── physics/                    # 9 Canonical PDE benchmark definitions
│   ├── base_pde.py             # Base PDE interface
│   ├── burgers.py              # 1D Viscous Burgers Shock
│   ├── convection2d.py         # 2D High-Péclet Convection
│   ├── quantum_schrodinger.py  # 2D Non-Linear Schrödinger (Quantum NLS)
│   ├── helmholtz.py            # 2D High-Frequency Helmholtz (k=4pi)
│   ├── high_dim_diffusion.py   # 4D Space + 1D Time Anisotropic Diffusion
│   ├── bratu2d.py              # 2D Non-Linear Bratu Thermal Ignition
│   ├── kovasznay.py            # 2D Steady Kovasznay Flow (Re=40)
│   ├── navier_stokes_cavity.py # 2D Lid-Driven Cavity Flow (Re=100)
│   └── klein_gordon.py         # 2D Non-Linear Relativistic Klein-Gordon
├── benchmarks/
│   ├── models_baseline.py      # Baseline models (Standard PINN, PCGrad, CAGrad)
│   └── run_master_benchmarks.py# Full 9-PDE automated master benchmark suite
├── results/
│   ├── data/                   # Benchmark CSV and JSON summary matrices
│   └── plots/                  # High-resolution comparison plots & territory maps
├── tests/                      # Automated unit test suite
├── train_two_stage_as_pinn.py  # Two-Stage Discover-and-Deploy training engine
├── kaggle_runner.ipynb         # One-click standalone Kaggle GPU deployment notebook
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/KartikeyaGangwar/as-pinn.git
cd as-pinn
pip install -r requirements.txt
```

### 2. Run All 9 Benchmarks (Master Suite)
```bash
python benchmarks/run_master_benchmarks.py
```

### 3. Run a Specific PDE Benchmark
```python
import torch
from physics.convection2d import HighPecletConvection2D
from train_two_stage_as_pinn import TwoStageASPINNTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pde = HighPecletConvection2D(device=device)
trainer = TwoStageASPINNTrainer(pde=pde, device=device, seed=42)

# Stage 1: Autonomous AMR Discovery
discovery_info = trainer.run_stage1_discovery(min_epochs=200, max_epochs=600)

# Stage 2: Production Training & Polish
model, history = trainer.train_stage2_production(
    discovered_info=discovery_info,
    hidden_dim=64,
    layers=4,
    adamw_epochs=1000,
    lbfgs_steps=250,
)
```

---

## 📄 License
This repository is open-sourced under the MIT License.
