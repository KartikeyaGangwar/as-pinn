# Adaptive $N$-Subspace Physics-Informed Neural Networks (AS-PINN)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 11.8 / 12.1](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--1973--7532-green.svg)](https://orcid.org/0009-0009-1973-7532)
[![Paper: Peer-Review](https://img.shields.io/badge/Status-Submission%20Ready-brightgreen.svg)](#citation)

**Official PyTorch Implementation** of the research paper:  
> **Adaptive $N$-Subspace Physics-Informed Neural Networks: Autonomous Parameter-Space AMR via Vectorized Gradient Conflict Profiling**  
> *Author:* **Kartikey Singh** ([ORCID: 0009-0009-1973-7532](https://orcid.org/0009-0009-1973-7532))  
> *Affiliation:* Department of Mathematics, University of Delhi, Delhi 110007, India  
> *Correspondence:* `kartikeysingh525@protonmail.com`  
> *Repository:* [https://github.com/KartikeyaGangwar/as-pinn](https://github.com/KartikeyaGangwar/as-pinn)

---

## 📌 Abstract & Scientific Overview

Physics-Informed Neural Networks (PINNs) have emerged as a cornerstone for solving forward and inverse partial differential equations (PDEs). However, monolithic deep neural architectures suffer from foundational optimization failure modes:
1. **Global Gradient Interference & Destruction:** Parameter updates across distinct spatial regions or boundary conditions conflict destructively ($\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$), causing optimization stagnation or boundary divergence.
2. **Spectral Bias & High-Frequency Attenuation:** Neural networks preferentially fit low-frequency modes, failing to capture high-frequency wave oscillations or thin shock layers.
3. **The Exponential Curse of Dimensionality:** Existing domain decomposition methods (e.g., XPINNs, FBPINNs) rely on rigid rectilinear grids requiring $\mathcal{O}(K^d)$ distinct networks, which becomes computationally intractable in $d \ge 4$ dimensions.

**AS-PINN** resolves these bottlenecks by introducing **Parameter-Space Adaptive Mesh Refinement (AMR)**:
- **Vectorized Gram Alignment Profiling:** Evaluates per-sample gradient alignment matrices $\mathcal{G}_{ij} = \cos \angle(\mathbf{g}_i, \mathbf{g}_j)$ using batched reverse-mode automatic differentiation (`torch.func.vmap`) with zero Python-loop overhead.
- **Autonomous Zero-Disruption Parameter Cleavage:** Automatically spawns and centers new localized parameter subspaces at regions of severe destructive gradient interference with provable solution invariance ($\|u^{(N+1)} - u^{(N)}\| \equiv 0$).
- **Dimension-Normalized Anisotropic Bandwidth Matrix:** Scales coordinate distance by dimension-specific bounds $\boldsymbol{\Sigma}_k = \operatorname{diag}(\sigma_1^2, \dots, \sigma_d^2)$, unblocking temporal cleavage in convection-dominated transport.
- **Two-Stage Discover-and-Deploy Workflow:** Discovers the optimal minimal subspace count $N^*$ and spatial centroids $\{\mathbf{c}_k\}$ using a lightweight probe in Stage 1, followed by clean production training and second-order Combined Global L-BFGS polish in Stage 2.

---

## 📊 Master 9-Benchmark Quantitative Matrix

All experiments were evaluated under standardized dense collocation meshes ($N_{\text{col}} = 10,000$), first-order AdamW with dynamic adaptive loss weighting, followed by full second-order Quasi-Newton Combined Global L-BFGS polish (Strong Wolfe line search).

| Benchmark System | Domain & PDE Type | Standard PINN | PCGrad (NeurIPS'20) | CAGrad (NeurIPS'21) | **AS-PINN (Ours)** | Discovered $N^*$ | Relative $L^2$ Error | AS-PINN Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Burgers 1D** | 1D Shock ($\nu = 0.01/\pi$) | $2.70 \times 10^{-1}$ | $6.10 \times 10^{0}$ | $1.03 \times 10^{1}$ | $\mathbf{5.37 \times 10^{-2}}$ | $N^*=7$ | $60.8\%$ | **$5.0\times$ lower loss** |
| **Convection 2D** | Advection ($\boldsymbol{\beta}=(10, 10)$) | $1.71 \times 10^{0}$ | $4.58 \times 10^{0}$ | $9.83 \times 10^{0}$ | $\mathbf{1.69 \times 10^{-2}}$ | $N^*=22$ | **$3.24\%$** | **$100.6\times$ lower loss** ($18.2\times$ better $L^2$) |
| **Quantum NLS** | 2D BEC Soliton | $1.74 \times 10^{-2}$ | $2.59 \times 10^{-1}$ | $4.07 \times 10^{1}$ | $\mathbf{1.19 \times 10^{-2}}$ | $N^*=21$ | $42.0\%$ | **$1.5\times$ lower loss** ($3409\times$ vs CAGrad) |
| **Helmholtz 2D** | High-Freq ($k=4\pi$) | $1.52 \times 10^{3}$ | $3.09 \times 10^{3}$ | $3.24 \times 10^{3}$ | $\mathbf{2.09 \times 10^{0}}$ | $N^*=21$ | $22.1\%$ | **$725.6\times$ lower loss** ($1551\times$ vs CAGrad) |
| **Diffusion 4D** | 4D Space + 1D Time | $7.52 \times 10^{-1}$ | $1.46 \times 10^{0}$ | $1.13 \times 10^{1}$ | $\mathbf{3.33 \times 10^{-1}}$ | $N^*=24$ | $33.5\%$ | **$2.3\times$ lower loss** ($34.0\times$ vs CAGrad) |
| **2D Bratu** | Thermal Runaway ($e^u$) | $1.18 \times 10^{-4}$ | $2.48 \times 10^{-3}$ | $2.81 \times 10^{1}$ | $\mathbf{1.56 \times 10^{-5}}$ | $N^*=17$ | **$0.0148\%$** | **$7.5\times$ lower loss** ($1.79\times 10^6\times$ vs CAGrad) |
| **Kovasznay** | Navier-Stokes Wake ($\mathrm{Re}=40$) | $2.97 \times 10^{-3}$ | $2.73 \times 10^{0}$ | $2.05 \times 10^{1}$ | $\mathbf{2.21 \times 10^{-4}}$ | $N^*=7$ | **$0.188\%$** | **$13.4\times$ lower loss** ($92783\times$ vs CAGrad) |
| **Cavity NS** | Incompressible ($\mathrm{Re}=100$) | $2.81 \times 10^{-1}$ | $2.56 \times 10^{-1}$ | $4.09 \times 10^{0}$ | $\mathbf{1.11 \times 10^{-1}}$ | $N^*=22$ | **$9.39\%$** | **$2.5\times$ lower loss** ($8.0\times$ better $L^2$) |
| **Klein-Gordon** | Relativistic Wave ($u^3$) | $1.69 \times 10^{0}$ | $2.67 \times 10^{1}$ | $4.02 \times 10^{3}$ | $\mathbf{1.86 \times 10^{-1}}$ | $N^*=21$ | **$1.94\%$** | **$9.1\times$ lower loss** ($21634\times$ vs CAGrad) |

---

## 🔬 Mathematical Formulation

The global solution $\hat{u}(\mathbf{x}): \Omega \to \mathbb{R}^{d_{\text{out}}}$ is parameterized as a smooth Partition of Unity (PoU) over $N$ localized parameter subspaces $\{\Phi_k(\cdot; \Theta_k)\}_{k=1}^N$:
$$\hat{u}(\mathbf{x}; \Theta) = \sum_{k=1}^N \psi_k(\mathbf{x}) \Phi_k\left(\boldsymbol{\Sigma}_k^{-1/2}(\mathbf{x} - \mathbf{c}_k); \Theta_k\right)$$

where the Partition of Unity gating satisfies $\sum_{k=1}^N \psi_k(\mathbf{x}) \equiv 1.0$ everywhere on $\Omega$:
$$\psi_k(\mathbf{x}) = \frac{\exp\left(-\frac{1}{2} (\mathbf{x} - \mathbf{c}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \mathbf{c}_k)\right)}{\sum_{j=1}^N \exp\left(-\frac{1}{2} (\mathbf{x} - \mathbf{c}_j)^T \boldsymbol{\Sigma}_j^{-1} (\mathbf{x} - \mathbf{c}_j)\right)} = \operatorname{softmax}_k \left( -\frac{1}{2} \sum_{i=1}^d \left(\frac{x_i - c_{k,i}}{\sigma_i}\right)^2 \right)$$

### Key Theoretical Guarantees:
1. **Zero-Disruption Cleavage Invariance (Theorem 2):**  
   At the instant of fission, allocating child subspace $\Phi_{N+1}$ initialized with parent weights $\Theta_p$ preserves the global solution and all derivatives identically ($\|u^{(N+1)} - u^{(N)}\|_{H^2(\Omega)} = 0$).
2. **Quasi-Block-Diagonal Hessian Conditioning (Theorem 3):**  
   Inter-subspace Hessian cross-coupling decays exponentially with spatial separation:
   $$\|\mathcal{E}_{jk}\|_2 \le C \cdot \exp\left(-\frac{\|\mathbf{c}_j - \mathbf{c}_k\|^2}{4\sigma^2}\right)$$
   enabling stable second-order Quasi-Newton optimization without interface tearing.
3. **Preservation of Green's Second Identity (Proposition 6):**  
   Because $\psi_k \in C^\infty(\Omega)$, no Dirac delta interface singularities or boundary jump energy penalties exist, exactly preserving Green's integral identities across subdomains.

---

## 💻 Hardware Architecture & Computational Runtime

All benchmark evaluations and AMR discovery runs were tested across two standardized environments:

| Parameter / Specification | Primary Local Testbed | Cloud GPU Accelerated Cluster |
| :--- | :--- | :--- |
| **Host Operating System** | Microsoft Windows 11 Enterprise (64-bit) | Linux Ubuntu 22.04 LTS (x86_64) |
| **Host CPU Processor** | Intel Core i5-9300H (4 Cores, 8 Threads @ 2.40 GHz) | Dual Intel Xeon vCPUs @ 2.20 GHz |
| **System Host RAM** | 16.0 GB DDR4 System Memory | 13.0 GB RAM |
| **Dedicated GPU Accelerator** | NVIDIA GeForce GTX 1650 (Turing SM 7.5) | NVIDIA Tesla T4 (16 GB) / P100 PCIe |
| **Dedicated GPU VRAM** | 4.0 GB GDDR6 | 16.0 GB GDDR6 / HBM2 |
| **NVIDIA Display Driver** | Release 610.88 (WDDM 3.2) | Release 535.104.05 |
| **CUDA Runtime Toolchain** | CUDA 11.8 (UMD 13.3) | CUDA 12.1 |
| **Deep Learning Framework** | PyTorch 2.7.1+cu118 | PyTorch 2.2.2+cu121 |
| **Vectorized Autodiff Engine** | `torch.func.vmap` (Batched Reverse-Mode AD) | `torch.func.vmap` |
| **Quasi-Newton Polish** | Combined Global L-BFGS (Strong Wolfe) | Combined Global L-BFGS (Strong Wolfe) |

---

## 🛠️ Repository Structure

```
Adaptive_N_Subspace_PINN/
├── models/
│   ├── as_pinn.py              # Core AdaptiveSubspacePINN & Symmetric Voronoi PoU
│   └── conflict_monitor.py     # Vectorized Gram conflict analyzer (torch.func.vmap)
├── physics/                    # 9 Canonical PDE benchmark definitions
│   ├── base_pde.py             # Abstract PDE base class
│   ├── burgers.py              # 1D Viscous Burgers Shock
│   ├── convection2d.py         # 2D High-Péclet Convection (beta=(10,10))
│   ├── quantum_schrodinger.py  # 2D Quantum Non-Linear Schrödinger Soliton
│   ├── helmholtz.py            # 2D High-Frequency Helmholtz (k=4pi)
│   ├── high_dim_diffusion.py   # 4D Space + 1D Time Anisotropic Diffusion
│   ├── bratu2d.py              # 2D Non-Linear Bratu Thermal Ignition
│   ├── kovasznay.py            # 2D Steady Kovasznay Flow (Re=40)
│   ├── navier_stokes_cavity.py # 2D Lid-Driven Cavity Flow (Re=100)
│   └── klein_gordon.py         # 2D Non-Linear Relativistic Klein-Gordon
├── benchmarks/
│   ├── models_baseline.py      # Baselines: Standard PINN, PCGrad, CAGrad
│   └── run_master_benchmarks.py# Full 9-PDE automated master benchmark suite
├── results/
│   ├── data/                   # master_benchmark_summary.csv and .json
│   └── plots/                  # Publication-ready 5-way comparisons & convergence grids
├── tests/                      # Automated unit test suite
├── train_two_stage_as_pinn.py  # Two-Stage Discover-and-Deploy training engine
├── kaggle_runner.ipynb         # Standalone one-click Kaggle deployment notebook
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies in a clean virtual environment:
```bash
git clone https://github.com/KartikeyaGangwar/as-pinn.git
cd as-pinn
pip install -r requirements.txt
```

### 2. Run All 9 PDE Benchmarks (Automated Suite)
Execute the complete comparative benchmark across all 9 canonical PDEs:
```bash
python benchmarks/run_master_benchmarks.py
```
Outputs are automatically saved into `results/data/master_benchmark_summary.csv` and `results/plots/`.

### 3. Run a Single Benchmark via Python API
```python
import torch
from physics.convection2d import HighPecletConvection2D
from train_two_stage_as_pinn import TwoStageASPINNTrainer

# Initialize GPU device and PDE formulation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pde = HighPecletConvection2D(device=device)

# Instantiate Two-Stage Trainer
trainer = TwoStageASPINNTrainer(pde=pde, device=device, seed=42)

# Stage 1: Autonomous AMR Discovery (Adaptive Quiescence)
discovery_info = trainer.run_stage1_discovery(min_epochs=200, max_epochs=600)
print(f"Discovered Subspaces: N* = {discovery_info['num_subspaces']}")

# Stage 2: Production Training & Second-Order Polish
model, history = trainer.train_stage2_production(
    discovered_info=discovery_info,
    hidden_dim=64,
    layers=4,
    adamw_epochs=1000,
    lbfgs_steps=250,
)
```

---

## 📖 Citation

If you use AS-PINN or our benchmarks in your research, please cite our manuscript:

```bibtex
@article{singh2026aspinn,
  title   = {Adaptive $N$-Subspace Physics-Informed Neural Networks: Autonomous Parameter-Space AMR via Vectorized Gradient Conflict Profiling},
  author  = {Singh, Kartikey},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/KartikeyaGangwar/as-pinn}
}
```

---

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


---

## 🏛️ The Adaptive Subspace (AS) Research Trilogy

This repository forms the central scientific milestone of a cohesive three-act theoretical research program authored by **Kartikey Singh**, systematically resolving destructive gradient interference across parameter manifolds:

```
                  THE ADAPTIVE SUBSPACE (AS) PARADIGM
                                   │
  ┌────────────────────────────────┼────────────────────────────────┐
  ▼                                ▼                                ▼
[ACT I: STATIC PHYSICAL]    [ACT II: DYNAMIC AMR]        [ACT III: FOUNDATIONAL CV]
Null-Space PINN             AS-PINN (This Repo)          AS-ViT
(Algebraic Direct-Sum)      (Autonomous PDE AMR)         (Feature-Space MoE)
[null-space-pinn]           [as-pinn]                    [as-vit-multitask]
DOI: 10.5281/zenodo.22132799                             Target: IEEE TPAMI / CVPR
```

1. **Act I: Algebraic Direct-Sum Partitioning (`null-space-pinn`):**  
   *Title:* *"On Parameter Decoupling, Conditioning, and Interface Transmission in Multi-Objective Physics-Informed Neural Networks"*  
   *Authors:* Kartikey Singh (University of Delhi) and Samarjeet Malik (IIT Jodhpur)  
   *Focus:* Proves structural gradient orthogonality ($\langle \nabla\mathcal{L}_{\mathrm{if}}, \nabla\mathcal{L}_{\mathrm{des}} \rangle \equiv 0$) on static material boundaries via $C^2$ Quintic Hermite operators and frozen orthogonal projection bases ($\Theta = \Theta_0 \oplus \Theta_1$, $\mathcal{W}_0 \mathcal{W}_1^T = \mathbf{0}$).  
   *Target:* Computer Methods in Applied Mechanics and Engineering (CMAME) | *Preprint DOI:* [10.5281/zenodo.22132799](https://doi.org/10.5281/zenodo.22132799) | *Repo:* [https://github.com/KartikeyaGangwar/null-space-pinn](https://github.com/KartikeyaGangwar/null-space-pinn)

2. **Act II: Autonomous Dynamic Parameter AMR (`as-pinn`):**  
   *Title:* *"Adaptive $N$-Subspace Physics-Informed Neural Networks: Autonomous Parameter-Space AMR via Vectorized Gradient Conflict Profiling"*  
   *Focus:* Generalizes static partitioning into dynamic, autonomous parameter-space Adaptive Mesh Refinement (AMR). Uses vectorized Gram conflict matrices (`torch.func.vmap`) to trigger autonomous subspace fission with exact zero-disruption solution invariance ($\|u^{(N+1)} - u^{(N)}\| \equiv 0$) across 9 canonical PDEs.  
   *Repo:* [https://github.com/KartikeyaGangwar/as-pinn](https://github.com/KartikeyaGangwar/as-pinn)

3. **Act III: Foundational Vision Transformers & MoE (`as-vit-multitask`):**  
   *Title:* *"AS-ViT: Adaptive Subspace Vision Transformers with Autonomous Gradient-Clash Routing for Multi-Task Learning"*  
   *Focus:* Transcends physical PDE space into the latent token manifold of Vision Transformers. Eliminates negative transfer in multi-task perception (segmentation, depth, surface normals) by continuously tracking inter-task Gram matrix eigenvalues ($\lambda_{\min}(\mathcal{G}) < -\tau_{\text{conflict}}$) and dynamically spawning expert subspaces via Feature-Conditioned Partition of Unity (PoU) gating.  
   *Repo:* [https://github.com/KartikeyaGangwar/as-vit-multitask](https://github.com/KartikeyaGangwar/as-vit-multitask) | *Target:* IEEE TPAMI / CVPR
