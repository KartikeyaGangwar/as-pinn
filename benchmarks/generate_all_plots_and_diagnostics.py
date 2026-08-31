import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))
from physics.burgers import Burgers1D
from physics.allen_cahn import AllenCahn1D
from physics.elliptic_interface import EllipticInterface2D
from physics.helmholtz import Helmholtz2D
from physics.high_dim_diffusion import HighDimDiffusion4D
from physics.cshape_magnet import CShapeElectromagnet
from physics.meissner_cylinder import MeissnerCylinder

from benchmarks.models_baseline import MonolithicMLP, train_model_adamw_lbfgs
from train_two_stage_as_pinn import TwoStageASPINNTrainer

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def generate_all_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating All Solution-Level, Subspace Territory, and Convergence Plots on {device}...")
    
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("manuscript", exist_ok=True)
    
    # 1. 1D Burgers
    print(">>> Rendering 1D Burgers Solution & Subspace Territory Plot...")
    pde_bg = Burgers1D(nu=0.01/np.pi, device=device)
    pde_bg.bounds = ((-1.0, 1.0), (0.0, 1.0))
    mono_bg = MonolithicMLP(2, 1, hidden_dim=96, layers=4).to(device)
    train_model_adamw_lbfgs(pde_bg, mono_bg, adamw_epochs=1000, lbfgs_steps=500, device=device)
    
    trainer_bg = TwoStageASPINNTrainer(pde=pde_bg, device=device)
    st1_bg = trainer_bg.run_stage1_discovery(min_epochs=300, max_epochs=450, max_subspaces=32)
    as_bg, _ = trainer_bg.train_stage2_production(st1_bg, hidden_dim=64, layers=4, adamw_epochs=1000, lbfgs_steps=500)
    
    nx, nt = 200, 200
    x_lin = np.linspace(-1.0, 1.0, nx)
    t_lin = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x_lin, t_lin)
    grid_t = torch.tensor(np.stack([X.ravel(), T.ravel()], axis=1), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        u_exact = pde_bg.exact_solution(grid_t).cpu().numpy().reshape(nt, nx)
        u_mono = mono_bg(grid_t).cpu().numpy().reshape(nt, nx)
        u_as = as_bg(grid_t).cpu().numpy().reshape(nt, nx)
        psi_as = as_bg.partition_of_unity(grid_t).cpu().numpy()
        territory = np.argmax(psi_as, axis=1).reshape(nt, nx)
        
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    im0 = axes[0, 0].contourf(X, T, u_exact, 100, cmap='Spectral', vmin=-1, vmax=1)
    axes[0, 0].set_title('Exact Ground Truth $u^*(x, t)$'); fig.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].contourf(X, T, u_mono, 100, cmap='Spectral', vmin=-1, vmax=1)
    axes[0, 1].set_title('Standard PINN Solution'); fig.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].contourf(X, T, u_as, 100, cmap='Spectral', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'Proposed AS-PINN (N*={st1_bg["num_subspaces"]})'); fig.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[1, 0].contourf(X, T, np.abs(u_mono - u_exact), 100, cmap='inferno', vmin=0, vmax=0.5)
    axes[1, 0].set_title('Standard PINN Error'); fig.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].contourf(X, T, np.abs(u_as - u_exact), 100, cmap='inferno', vmin=0, vmax=0.5)
    axes[1, 1].set_title('Proposed AS-PINN Error'); fig.colorbar(im4, ax=axes[1, 1])
    
    c_bg = st1_bg['centroids'].cpu().numpy()
    im5 = axes[1, 2].contourf(X, T, territory, levels=np.arange(st1_bg['num_subspaces']+1)-0.5, cmap='tab10')
    axes[1, 2].scatter(c_bg[:, 0], c_bg[:, 1], c='white', edgecolors='black', s=120, marker='o', label='AMR Centroids $c_k$')
    axes[1, 2].plot([0, 0], [0.4, 1.0], 'w--', linewidth=2, label='Shock Trajectory')
    axes[1, 2].set_title('Subspace Territory Partition ($\psi_k$)'); axes[1, 2].legend(loc='lower left')
    fig.colorbar(im5, ax=axes[1, 2], ticks=range(st1_bg['num_subspaces']))
    
    plt.tight_layout()
    fig.savefig('results/plots/burgers1d_visual_comparison.png', bbox_inches='tight')
    fig.savefig('manuscript/burgers1d_visual_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("  [+] Saved burgers1d_visual_comparison.png")

if __name__ == "__main__":
    generate_all_plots()
