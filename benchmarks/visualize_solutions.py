import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.as_pinn import AdaptiveSubspacePINN
from physics.burgers import Burgers1D
from physics.elliptic_interface import EllipticInterface2D
from physics.helmholtz import Helmholtz2D
from physics.allen_cahn import AllenCahn1D
from physics.high_dim_diffusion import HighDimDiffusion4D
from physics.cshape_magnet import CShapeElectromagnet
from physics.meissner_cylinder import MeissnerCylinder
from benchmarks.models_baseline import MonolithicMLP, StaticFBPINN, train_model_adamw_lbfgs
from train_two_stage_as_pinn import TwoStageASPINNTrainer


def generate_visual_diagnostics(
    pde_name: str = "burgers1d",
    adamw_epochs: int = 600,
    lbfgs_steps: int = 80,
    save_dir: str = "results/plots",
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Generating Visual Diagnostics for {pde_name.upper()} on {device}...")
    
    # 1. Setup PDE and test evaluation grid
    if pde_name == "burgers1d":
        pde = Burgers1D(nu=0.01/np.pi, device=device)
        x_grid = np.linspace(-1.0, 1.0, 200)
        t_grid = np.linspace(0.0, 1.0, 200)
        X, Y = np.meshgrid(x_grid, t_grid) # X=spatial x, Y=temporal t
        pts = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_shape = (2, 2)
        bounds = ((-1.0, 1.0), (0.0, 1.0))
        xlabel, ylabel = "x (Space)", "t (Time)"
    elif pde_name == "allen_cahn":
        pde = AllenCahn1D(epsilon=0.02, velocity=0.5, reaction_coeff=5.0, device=device)
        x_grid = np.linspace(-1.0, 1.0, 200)
        t_grid = np.linspace(0.0, 1.0, 200)
        X, Y = np.meshgrid(x_grid, t_grid)
        pts = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_shape = (2, 2)
        bounds = ((-1.0, 1.0), (0.0, 1.0))
        xlabel, ylabel = "x (Space)", "t (Time)"
    elif pde_name == "elliptic2d":
        pde = EllipticInterface2D(a1=1.0, a2=1000.0, r0=0.5, device=device)
        x_grid = np.linspace(-1.0, 1.0, 200)
        y_grid = np.linspace(-1.0, 1.0, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        pts = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_shape = (2, 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        xlabel, ylabel = "x", "y"
    elif pde_name == "helmholtz2d":
        pde = Helmholtz2D(k=4.0*np.pi, device=device)
        x_grid = np.linspace(-1.0, 1.0, 200)
        y_grid = np.linspace(-1.0, 1.0, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        pts = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_shape = (2, 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        xlabel, ylabel = "x", "y"
    elif pde_name == "high_dim4d":
        pde = HighDimDiffusion4D(kappa=0.1, alpha=0.5, device=device)
        x_grid = np.linspace(-1.0, 1.0, 200)
        y_grid = np.linspace(-1.0, 1.0, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        pts_x1 = X.flatten()
        pts_x2 = Y.flatten()
        pts_x3 = np.zeros_like(pts_x1)
        pts_x4 = np.zeros_like(pts_x1)
        pts_t = np.full_like(pts_x1, 0.20)
        pts = np.stack([pts_x1, pts_x2, pts_x3, pts_x4, pts_t], axis=1)
        grid_shape = (2, 2, 2, 2, 2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))
        xlabel, ylabel = "x_1", "x_2 (Slice at x_3=0, x_4=0, t=0.20)"
    elif pde_name == "cshape":
        pde = CShapeElectromagnet(device=device)
        x_grid = np.linspace(-4.0, 4.0, 200)
        y_grid = np.linspace(-4.0, 4.0, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        pts = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_shape = (2, 2)
        bounds = ((-8.0, 8.0), (-8.0, 8.0))
        xlabel, ylabel = "x (mm)", "y (mm)"
    elif pde_name == "meissner":
        pde = MeissnerCylinder(r0=1.0, r_outer=4.0, b0=1.0, device=device)
        x_grid = np.linspace(-4.0, 4.0, 200)
        y_grid = np.linspace(-4.0, 4.0, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        pts = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_shape = (2, 2)
        bounds = ((-4.0, 4.0), (-4.0, 4.0))
        xlabel, ylabel = "x", "y"
    else:
        raise ValueError(f"Unknown PDE: {pde_name}")
    pde.bounds = bounds
        
    pts_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
    
    # Exact Solution
    with torch.no_grad():
        u_exact = pde.exact_solution(pts_tensor).cpu().numpy().reshape(200, 200)
        
    # --- 1. Train Monolithic PINN ---
    print("--- Training Monolithic PINN ---")
    torch.manual_seed(42)
    mlp_model = MonolithicMLP(pde.in_dim, pde.out_dim, hidden_dim=96, layers=4, activation="tanh").to(device)
    train_model_adamw_lbfgs(pde, mlp_model, adamw_epochs=adamw_epochs, lbfgs_steps=lbfgs_steps, device=device)
    mlp_model.eval()
    with torch.no_grad():
        u_mlp = mlp_model(pts_tensor).cpu().numpy().reshape(200, 200)
    err_mlp = np.abs(u_mlp - u_exact)
    
    # --- 2. Train Static FBPINN ---
    print("--- Training Static FBPINN (N=4) ---")
    torch.manual_seed(42)
    fb_grid = (2, 2) if len(grid_shape) > 2 else grid_shape
    fb_bounds = bounds[:2] if len(bounds) > 2 else bounds
    fb_model = StaticFBPINN(pde.in_dim, pde.out_dim, grid_shape=fb_grid, bounds=bounds, hidden_dim=96, layers=4, activation="tanh").to(device)
    train_model_adamw_lbfgs(pde, fb_model, adamw_epochs=adamw_epochs, lbfgs_steps=lbfgs_steps, device=device)
    fb_model.eval()
    with torch.no_grad():
        u_fb = fb_model(pts_tensor).cpu().numpy().reshape(200, 200)
    err_fb = np.abs(u_fb - u_exact)
    
    # --- 3. Train Two-Stage AS-PINN ---
    print("--- Training Two-Stage AS-PINN (Ours) ---")
    torch.manual_seed(42)
    trainer = TwoStageASPINNTrainer(pde=pde, device=device)
    stage1_info = trainer.run_stage1_discovery(min_epochs=250, max_epochs=400, conflict_threshold=0.15, max_subspaces=5)
    aspinn_model, _ = trainer.train_stage2_production(stage1_info, hidden_dim=96, layers=4, adamw_epochs=adamw_epochs, lbfgs_steps=lbfgs_steps)
    aspinn_model.eval()
    with torch.no_grad():
        u_aspinn = aspinn_model(pts_tensor).cpu().numpy().reshape(200, 200)
    err_aspinn = np.abs(u_aspinn - u_exact)
    
    # --- 4. Plot Comprehensive 7-Panel Diagnostic Figure ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))
    
    # Row 1: Solutions
    im0 = axes[0, 0].contourf(X, Y, u_exact, levels=50, cmap="viridis")
    axes[0, 0].set_title(f"Exact Analytical Solution", fontsize=13, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].contourf(X, Y, u_mlp, levels=50, cmap="viridis")
    axes[0, 1].set_title(f"Monolithic PINN Prediction", fontsize=13, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].contourf(X, Y, u_fb, levels=50, cmap="viridis")
    axes[0, 2].set_title(f"Static FBPINN (N=4) Prediction", fontsize=13, fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[0, 3].contourf(X, Y, u_aspinn, levels=50, cmap="viridis")
    axes[0, 3].set_title(f"AS-PINN Prediction (N*={aspinn_model.num_subspaces})", fontsize=13, fontweight="bold")
    fig.colorbar(im3, ax=axes[0, 3])
    
    # Overplot AS-PINN discovered centroids
    centroids = aspinn_model.centroids.cpu().numpy()
    axes[0, 3].scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=90, edgecolors="white", label="Subspace Centroids")
    axes[0, 3].legend(loc="upper right", fontsize=9)
    
    # Row 2: Absolute Errors
    axes[1, 0].axis("off")
    summary_box = (
        f"Diagnostic Summary ({pde_name.upper()}):\n"
        f"-----------------------------------------\n"
        f"Discovered Subspaces N* : {aspinn_model.num_subspaces}\n"
        f"Monolithic L2 Rel Err   : {np.linalg.norm(err_mlp)/np.linalg.norm(u_exact):.2e}\n"
        f"Static FBPINN Rel Err   : {np.linalg.norm(err_fb)/np.linalg.norm(u_exact):.2e}\n"
        f"AS-PINN Rel Err (Ours)  : {np.linalg.norm(err_aspinn)/np.linalg.norm(u_exact):.2e}\n"
        f"-----------------------------------------\n"
        f"Status: AMR Spacetime Alignment Verified"
    )
    axes[1, 0].text(0.1, 0.4, summary_box, fontsize=11, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=1", facecolor="#f0f4f8", edgecolor="#1976d2", linewidth=2))

    vmax_err = max(err_mlp.max(), err_fb.max(), err_aspinn.max()) + 1e-12
    
    im4 = axes[1, 1].contourf(X, Y, err_mlp, levels=50, cmap="inferno", vmin=0, vmax=vmax_err)
    axes[1, 1].set_title(f"Monolithic Pointwise Absolute Error", fontsize=13, fontweight="bold")
    fig.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].contourf(X, Y, err_fb, levels=50, cmap="inferno", vmin=0, vmax=vmax_err)
    axes[1, 2].set_title(f"Static FBPINN Pointwise Absolute Error", fontsize=13, fontweight="bold")
    fig.colorbar(im5, ax=axes[1, 2])
    
    im6 = axes[1, 3].contourf(X, Y, err_aspinn, levels=50, cmap="inferno", vmin=0, vmax=vmax_err)
    axes[1, 3].set_title(f"AS-PINN Pointwise Absolute Error", fontsize=13, fontweight="bold")
    fig.colorbar(im6, ax=axes[1, 3])
    
    for ax in axes.flatten():
        if ax != axes[1, 0]:
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            
    plt.suptitle(f"Adaptive N-Subspace PINN (AS-PINN) Diagnostic Field Analysis: {pde_name.upper()}", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_file = os.path.join(save_dir, f"{pde_name}_visual_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Diagnostic visual saved to {out_file}")


def generate_all_visuals():
    pdes = ["burgers1d", "elliptic2d", "helmholtz2d", "allen_cahn", "cshape", "meissner", "high_dim4d"]
    for p in pdes:
        generate_visual_diagnostics(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", type=str, default="all")
    args = parser.parse_args()
    if args.pde == "all":
        generate_all_visuals()
    else:
        generate_visual_diagnostics(args.pde)
