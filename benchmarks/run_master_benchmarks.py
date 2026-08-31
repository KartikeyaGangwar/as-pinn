import os
import sys
import time
import json
import csv
import copy
import gc
import shutil
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.abspath('.'))
from physics.burgers import Burgers1D
from physics.convection2d import HighPecletConvection2D
from physics.quantum_schrodinger import QuantumNonlinearSchrodinger2D
from physics.helmholtz import Helmholtz2D
from physics.high_dim_diffusion import HighDimDiffusion4D
from physics.bratu2d import NonLinearBratu2D
from physics.kovasznay import KovasznayFlow2D
from physics.navier_stokes_cavity import LidDrivenCavityNavierStokes
from physics.klein_gordon import KleinGordon2D

from benchmarks.models_baseline import (
    MonolithicMLP,
    train_model_adamw_lbfgs,
    train_pcgrad_baseline,
    train_cagrad_baseline,
)
from train_two_stage_as_pinn import TwoStageASPINNTrainer, cleanup_gpu, set_seed

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Bitstream Vera Serif', 'Computer Modern Roman', 'serif'],
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'axes.formatter.use_mathtext': True,
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

def compute_model_losses(model, pde, n_pts=4096):
    model.eval()
    x_int = pde.sample_interior(n_pts)
    res = pde.compute_residuals(model, x_int)
    l_res = torch.mean(res ** 2).item()
    
    x_bc, u_bc = pde.sample_boundary(1024)
    l_bc = pde.compute_boundary_loss(model, x_bc, u_bc).item()
    
    l_ic = 0.0
    ic_s = pde.sample_initial(1024)
    if ic_s is not None:
        x_ic, u_ic = ic_s
        l_ic = pde.compute_initial_loss(model, x_ic, u_ic).item()
        
    tot = l_res + 20.0 * l_bc + 20.0 * l_ic
    
    rel_l2 = 0.0
    if hasattr(pde, "compute_relative_l2_error"):
        try:
            rel_l2 = pde.compute_relative_l2_error(model, n_test=n_pts)
        except Exception:
            rel_l2 = 0.0
            
    return tot, l_res, l_bc, l_ic, rel_l2


def plot_individual_convergence(pde_key, pde_name, hist_pinn, hist_pcg, hist_cag, hist_as, save_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if hist_pinn and "loss" in hist_pinn and len(hist_pinn["loss"]) > 0:
        n_p = min(len(hist_pinn["epoch"]), len(hist_pinn["loss"]))
        ax.semilogy(hist_pinn["epoch"][:n_p], hist_pinn["loss"][:n_p], label=r"$\mathbf{Standard\ PINN}\ (\mathrm{Raissi\ et\ al.})$", color="#1f77b4", linewidth=2.0)
    if hist_pcg and "loss" in hist_pcg and len(hist_pcg["loss"]) > 0:
        n_pc = min(len(hist_pcg["epoch"]), len(hist_pcg["loss"]))
        ax.semilogy(hist_pcg["epoch"][:n_pc], hist_pcg["loss"][:n_pc], label=r"$\mathbf{PCGrad}\ (\mathrm{Yu\ et\ al.})$", color="#ff7f0e", linewidth=2.0)
    if hist_cag and "loss" in hist_cag and len(hist_cag["loss"]) > 0:
        n_ca = min(len(hist_cag["epoch"]), len(hist_cag["loss"]))
        ax.semilogy(hist_cag["epoch"][:n_ca], hist_cag["loss"][:n_ca], label=r"$\mathbf{CAGrad}\ (\mathrm{Liu\ et\ al.})$", color="#2ca02c", linewidth=2.0)
    if hist_as and "loss_total" in hist_as and len(hist_as["loss_total"]) > 0:
        n_as = min(len(hist_as["epoch"]), len(hist_as["loss_total"]))
        ax.semilogy(hist_as["epoch"][:n_as], hist_as["loss_total"][:n_as], label=r"$\mathbf{Proposed\ AS-PINN}\ (\mathrm{Ours})$", color="#d62728", linewidth=2.5)
        
    ax.set_title(f"Total Loss Convergence $\\mathcal{{L}}(\\Theta)$: {pde_name}", fontweight="bold", pad=12)
    ax.set_xlabel("Optimization Epochs / Evaluations (AdamW + L-BFGS Polish)")
    ax.set_ylabel(r"$\mathrm{Total\ Loss\ } \mathcal{L}(\Theta)\ [\mathrm{Log\ Scale}]$")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc")
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    [+] Saved Convergence Plot: {save_path}")


def plot_solutions_comparison(pde_key, pde_name, pde, pinn, pcg, cag, aspinn, save_path):
    device = pde.device
    nx, ny = 220, 220
    is_bipolar = False
    draw_geom_func = None
    
    if pde_key == "burgers1d":
        bounds = pde.bounds if hasattr(pde, 'bounds') else ((-1, 1), (0, 1))
        X, Y = np.meshgrid(np.linspace(bounds[0][0], bounds[0][1], nx), np.linspace(bounds[1][0], bounds[1][1], ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$t$"
        cmap = "turbo"
    elif pde_key == "convection2d":
        two_pi = float(2.0 * np.pi)
        X, Y = np.meshgrid(np.linspace(0.0, two_pi, nx), np.linspace(0.0, two_pi, ny))
        t_c = 0.50 * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        cmap = "coolwarm"
        is_bipolar = True
    elif pde_key == "quantum_nls":
        X, Y = np.meshgrid(np.linspace(-3.0, 3.0, nx), np.linspace(-3.0, 3.0, ny))
        t_c = float(np.pi / 4.0) * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        cmap = "magma"
    elif pde_key == "helmholtz2d":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        cmap = "viridis"
    elif pde_key == "high_dim4d":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        zeros = np.zeros_like(X.ravel())
        t_c = 0.20 * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), zeros, zeros, t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x_1$", "$x_2$"
        cmap = "magma"
    elif pde_key == "bratu2d":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        cmap = "inferno"
    elif pde_key == "kovasznay":
        X, Y = np.meshgrid(np.linspace(-0.5, 1.0, nx), np.linspace(-0.5, 1.5, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        cmap = "viridis"
    elif pde_key == "cavity_ns":
        X, Y = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        cmap = "plasma"
        def draw_geom(ax):
            ax.add_patch(patches.Rectangle((0, 0), 1.0, 1.0, edgecolor='black', facecolor='none', linewidth=1.5))
        draw_geom_func = draw_geom
    elif pde_key == "klein_gordon":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        t_c = 0.50 * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        cmap = "RdBu_r"
        is_bipolar = True
        
    with torch.no_grad():
        u_exact = pde.exact_solution(pts).cpu().numpy() if hasattr(pde, "exact_solution") else None
        if u_exact is not None:
            u_exact = u_exact[:, 0:1].reshape(ny, nx) if u_exact.shape[1] > 1 else u_exact.reshape(ny, nx)
        
        if pde_key == "quantum_nls":
            pinn_out = pinn(pts)
            u_pinn = (pinn_out[:, 0:1]**2 + pinn_out[:, 1:2]**2).cpu().numpy().reshape(ny, nx)
            pcg_out = pcg(pts)
            u_pcg = (pcg_out[:, 0:1]**2 + pcg_out[:, 1:2]**2).cpu().numpy().reshape(ny, nx)
            cag_out = cag(pts)
            u_cag = (cag_out[:, 0:1]**2 + cag_out[:, 1:2]**2).cpu().numpy().reshape(ny, nx)
            as_out = aspinn(pts)
            u_as = (as_out[:, 0:1]**2 + as_out[:, 1:2]**2).cpu().numpy().reshape(ny, nx)
        elif pde_key in ["cavity_ns", "kovasznay"]:
            u_pinn = pinn(pts)[:, 0:1].cpu().numpy().reshape(ny, nx)
            u_pcg = pcg(pts)[:, 0:1].cpu().numpy().reshape(ny, nx)
            u_cag = cag(pts)[:, 0:1].cpu().numpy().reshape(ny, nx)
            u_as = aspinn(pts)[:, 0:1].cpu().numpy().reshape(ny, nx)
        else:
            u_pinn = pinn(pts)[:, 0:1].cpu().numpy().reshape(ny, nx) if pinn(pts).shape[1] > 1 else pinn(pts).cpu().numpy().reshape(ny, nx)
            u_pcg = pcg(pts)[:, 0:1].cpu().numpy().reshape(ny, nx) if pcg(pts).shape[1] > 1 else pcg(pts).cpu().numpy().reshape(ny, nx)
            u_cag = cag(pts)[:, 0:1].cpu().numpy().reshape(ny, nx) if cag(pts).shape[1] > 1 else cag(pts).cpu().numpy().reshape(ny, nx)
            u_as = aspinn(pts)[:, 0:1].cpu().numpy().reshape(ny, nx) if aspinn(pts).shape[1] > 1 else aspinn(pts).cpu().numpy().reshape(ny, nx)
        
    ref_u = u_exact if u_exact is not None else u_as
    
    if is_bipolar:
        vmax = float(np.percentile(np.abs(ref_u), 98))
        if vmax < 1e-6: vmax = 1.0
        vmin = -vmax
    else:
        vmin, vmax = float(np.min(ref_u)), float(np.max(ref_u))
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4), sharey=True)
    models_data = [
        (r"$\mathbf{Ground\ Truth\ } u^*(\mathbf{x})$" if u_exact is not None else r"$\mathbf{Analytical\ Reference}$", u_exact if u_exact is not None else u_as),
        (r"$\mathbf{Standard\ PINN}$", u_pinn),
        (r"$\mathbf{PCGrad\ Baseline}$", u_pcg),
        (r"$\mathbf{CAGrad\ Baseline}$", u_cag),
        (r"$\mathbf{Proposed\ AS-PINN\ (Ours)}$", u_as),
    ]
    
    n_levels = 120 if pde_key == "burgers1d" else 100
    for i, (title, u_data) in enumerate(models_data):
        im = axes[i].contourf(X, Y, u_data, n_levels, cmap=cmap, vmin=vmin, vmax=vmax)
        if is_bipolar or pde_key in ["meissner", "cavity_ns", "quantum_nls"]:
            levels = np.linspace(vmin, vmax, 16)
            axes[i].contour(X, Y, u_data, levels=levels, colors='black', linewidths=0.55, alpha=0.4)
        if draw_geom_func is not None:
            draw_geom_func(axes[i])
        axes[i].set_title(title, fontweight="bold", fontsize=11, pad=8)
        axes[i].set_xlabel(x_label)
        if i == 0:
            axes[i].set_ylabel(y_label)
            
    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    field_label = r"$\mathrm{Probability\ Density\ } |\psi|^2$" if pde_key == "quantum_nls" else r"$\mathrm{Field\ Solution\ } u$"
    cbar.set_label(field_label, fontweight="bold")
    
    fig.suptitle(f"{pde_name} — 5-Way Solution Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    [+] Saved Solutions Comparison: {save_path}")


def plot_subspace_territory_map(pde_key, pde_name, pde, aspinn, stage1_info, save_path):
    device = pde.device
    nx, ny = 220, 220
    num_s = stage1_info["num_subspaces"]
    centroids = stage1_info["centroids"].cpu().numpy()
    sigma = stage1_info["bandwidth"]
    
    if pde_key == "burgers1d":
        bounds = ((-1, 1), (0, 1))
        X, Y = np.meshgrid(np.linspace(bounds[0][0], bounds[0][1], nx), np.linspace(bounds[1][0], bounds[1][1], ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$t$"
    elif pde_key == "convection2d":
        two_pi = float(2.0 * np.pi)
        X, Y = np.meshgrid(np.linspace(0.0, two_pi, nx), np.linspace(0.0, two_pi, ny))
        t_c = 0.50 * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
    elif pde_key == "quantum_nls":
        X, Y = np.meshgrid(np.linspace(-3.0, 3.0, nx), np.linspace(-3.0, 3.0, ny))
        t_c = float(np.pi / 4.0) * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
    elif pde_key == "helmholtz2d":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
    elif pde_key == "high_dim4d":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        zeros = np.zeros_like(X.ravel())
        t_c = 0.20 * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), zeros, zeros, t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x_1$", "$x_2$"
    elif pde_key == "bratu2d":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
    elif pde_key == "kovasznay":
        X, Y = np.meshgrid(np.linspace(-0.5, 1.0, nx), np.linspace(-0.5, 1.5, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
    elif pde_key == "cavity_ns":
        X, Y = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
    elif pde_key == "klein_gordon":
        X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
        t_c = 0.50 * np.ones_like(X.ravel())
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        x_label, y_label = "$x$", "$y$"
        
    with torch.no_grad():
        c_t = stage1_info["centroids"].to(device)
        sigma_t = stage1_info["bandwidth"]
        if not isinstance(sigma_t, torch.Tensor):
            sigma_t = torch.tensor(sigma_t, device=device, dtype=torch.float32)
        else:
            sigma_t = sigma_t.to(device=device, dtype=torch.float32)
        sigma_t = sigma_t.reshape(1, 1, -1)
        diff = (pts.unsqueeze(1) - c_t.unsqueeze(0)) / sigma_t
        dist_sq = torch.sum(diff ** 2, dim=-1)
        logits = -0.5 * dist_sq
        psi = torch.softmax(logits, dim=-1).cpu().numpy()
        territory = np.argmax(psi, axis=1).reshape(ny, nx)
        
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap_terr = plt.colormaps.get_cmap("tab10" if num_s <= 10 else "tab20")
    im = ax.contourf(X, Y, territory, levels=np.arange(num_s + 1) - 0.5, cmap=cmap_terr)
    if centroids.shape[1] >= 2:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="white", edgecolors="black", s=130, marker="o", linewidths=1.8, label=r"$\mathrm{Discovered\ Centroids\ } \mathbf{c}_k$", zorder=5)
        
    ax.set_title(f"Discovered Subspace Territories: {pde_name} ($N^*={num_s}$)", fontweight="bold", pad=12)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc")
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    [+] Saved Subspace Territory Map: {save_path}")


def plot_pou_overlap_profile(pde_key, pde_name, pde, aspinn, stage1_info, save_path):
    device = pde.device
    num_s = stage1_info["num_subspaces"]
    n_pts = 400
    
    if pde_key == "burgers1d":
        s = np.linspace(-1.0, 1.0, n_pts)
        pts = torch.tensor(np.stack([s, 0.5 * np.ones_like(s)], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Spatial 1D Profile along $x$ (at fixed $t=0.50$)"
    elif pde_key == "convection2d":
        two_pi = float(2.0 * np.pi)
        s = np.linspace(0.0, two_pi, n_pts)
        t_c = 0.50 * np.ones_like(s)
        pts = torch.tensor(np.stack([s, s, t_c], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Diagonal 1D Profile along $(x, y=x, t=0.50)$"
    elif pde_key == "quantum_nls":
        s = np.linspace(-3.0, 3.0, n_pts)
        t_c = float(np.pi / 4.0) * np.ones_like(s)
        pts = torch.tensor(np.stack([s, np.zeros_like(s), t_c], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Spatial 1D Profile along $x$ (at $y=0, t=\pi/4$)"
    elif pde_key == "helmholtz2d":
        s = np.linspace(-1.0, 1.0, n_pts)
        pts = torch.tensor(np.stack([s, s], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Diagonal 1D Profile along $(x, y = x)$"
    elif pde_key == "high_dim4d":
        s = np.linspace(-1.0, 1.0, n_pts)
        zeros = np.zeros_like(s)
        t_c = 0.20 * np.ones_like(s)
        pts = torch.tensor(np.stack([s, s, zeros, zeros, t_c], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Hyper-Diagonal 1D Profile $(x_1, x_2 = x_1, 0, 0, t=0.20)$"
    elif pde_key == "bratu2d":
        s = np.linspace(-1.0, 1.0, n_pts)
        pts = torch.tensor(np.stack([s, np.zeros_like(s)], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Centerline 1D Profile along $x$ (at $y=0$)"
    elif pde_key == "kovasznay":
        s = np.linspace(-0.5, 1.0, n_pts)
        pts = torch.tensor(np.stack([s, np.zeros_like(s)], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Centerline 1D Profile along $x$ (at $y=0$)"
    elif pde_key == "cavity_ns":
        s = np.linspace(0.0, 1.0, n_pts)
        pts = torch.tensor(np.stack([s, 0.5 * np.ones_like(s)], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Cavity Centerline 1D Profile along $x$ (at $y=0.50$)"
    elif pde_key == "klein_gordon":
        s = np.linspace(-1.0, 1.0, n_pts)
        t_c = 0.50 * np.ones_like(s)
        pts = torch.tensor(np.stack([s, s, t_c], axis=1), dtype=torch.float32, device=device)
        cut_label = r"Diagonal 1D Profile along $(x, y=x, t=0.50)$"
        
    with torch.no_grad():
        c_t = stage1_info["centroids"].to(device)
        sigma_t = stage1_info["bandwidth"]
        if not isinstance(sigma_t, torch.Tensor):
            sigma_t = torch.tensor(sigma_t, device=device, dtype=torch.float32)
        else:
            sigma_t = sigma_t.to(device=device, dtype=torch.float32)
        sigma_t = sigma_t.reshape(1, 1, -1)
        diff = (pts.unsqueeze(1) - c_t.unsqueeze(0)) / sigma_t
        dist_sq = torch.sum(diff ** 2, dim=-1)
        logits = -0.5 * dist_sq
        psi_vals = torch.softmax(logits, dim=-1).cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cmap_p = plt.colormaps.get_cmap("tab10" if num_s <= 10 else "tab20")
    for k in range(num_s):
        ax.plot(s, psi_vals[:, k], label=rf"$\psi_{{{k+1}}}(\mathbf{{x}})$", color=cmap_p(k % 20), linewidth=1.8)
    ax.plot(s, np.sum(psi_vals, axis=1), "k--", linewidth=2.2, label=r"$\sum_k \psi_k(\mathbf{x}) \equiv 1$ (Exact PoU)")
    ax.set_title(f"Smooth Partition of Unity Overlap Profile: {pde_name}", fontweight="bold", pad=12)
    ax.set_xlabel(cut_label); ax.set_ylabel(r"$\mathrm{Partition\ Weight\ } \psi_k(\mathbf{x})$")
    ax.set_ylim([-0.05, 1.15]); ax.grid(True, linestyle="--", alpha=0.4)
    ncols = min(6, num_s + 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncols, frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    [+] Saved PoU Overlap Profile: {save_path}")


def run_master_suite(custom_benchmarks=None):
    print("="*105)
    print("      ADAPTIVE N-SUBSPACE PINN (AS-PINN) — 9-PDE MASTER BENCHMARK SUITE")
    print("="*105)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} | CUDA Available: {torch.cuda.is_available()}")
    
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("manuscript", exist_ok=True)
    
    ALL_BENCHMARKS = custom_benchmarks or [
        ("burgers1d", "Burgers 1D Shock", Burgers1D, {"epochs": 1000, "lbfgs": 250, "n_int": 8192}),
        ("convection2d", "2D High-Peclet Convection (beta=10)", HighPecletConvection2D, {"epochs": 1500, "lbfgs": 350, "n_int": 10240}),
        ("quantum_nls", "2D Non-Linear Schrodinger (Quantum NLS)", QuantumNonlinearSchrodinger2D, {"epochs": 1500, "lbfgs": 350, "n_int": 10240}),
        ("helmholtz2d", "2D High-Frequency Helmholtz (k=4pi)", Helmholtz2D, {"epochs": 2500, "lbfgs": 500, "n_int": 10240}),
        ("high_dim4d", "4D Anisotropic Diffusion", HighDimDiffusion4D, {"epochs": 2500, "lbfgs": 500, "n_int": 30000}),
        ("bratu2d", "2D Non-Linear Bratu Thermal Ignition", NonLinearBratu2D, {"epochs": 1500, "lbfgs": 350, "n_int": 10240}),
        ("kovasznay", "2D Kovasznay Flow (Re=40)", KovasznayFlow2D, {"epochs": 1500, "lbfgs": 350, "n_int": 10240}),
        ("cavity_ns", "2D Cavity Navier-Stokes (Re=100)", LidDrivenCavityNavierStokes, {"epochs": 1500, "lbfgs": 350, "n_int": 10240}),
        ("klein_gordon", "2D Klein-Gordon Relativistic Wave", KleinGordon2D, {"epochs": 1500, "lbfgs": 350, "n_int": 10240}),
    ]
    
    results = []
    subspace_summary_records = {}
    
    for idx, (pde_key, pde_name, pde_cls, pde_cfg) in enumerate(ALL_BENCHMARKS, 1):
        print("\n" + "="*105)
        print(f"[{idx}/9] STARTING BENCHMARK: {pde_name.upper()} ({pde_key})")
        print(f"       Configuration: AdamW Epochs = {pde_cfg['epochs']} | L-BFGS Steps = {pde_cfg['lbfgs']} | Collocation = {pde_cfg['n_int']}")
        print("="*105)
        
        cleanup_gpu()
        pde = pde_cls(device=device)
        entry = {"benchmark": pde_name, "pde_key": pde_key}
        ep = pde_cfg["epochs"]
        lbfgs_s = pde_cfg["lbfgs"]
        
        # 1. Standard PINN
        print(f"\n  [1/4] Training Standard PINN (Raissi et al.)...")
        cleanup_gpu()
        pinn_model = MonolithicMLP(pde.in_dim, pde.out_dim, hidden_dim=96, layers=4).to(device)
        t0 = time.time()
        hist_pinn = train_model_adamw_lbfgs(pde, pinn_model, adamw_epochs=ep, lbfgs_steps=lbfgs_s, device=device)
        t_pinn = time.time() - t0
        tot_pinn, res_pinn, bc_pinn, _, rel_pinn = compute_model_losses(pinn_model, pde)
        entry["pinn_loss_total"] = tot_pinn; entry["pinn_loss_res"] = res_pinn; entry["pinn_loss_bc"] = bc_pinn; entry["pinn_rel_l2"] = rel_pinn; entry["pinn_time"] = t_pinn
        print(f"    Standard PINN -> Total Loss: {tot_pinn:.6e} | Res: {res_pinn:.6e} | Rel L2: {rel_pinn:.4e} | Time: {t_pinn:.1f}s")
        
        # 2. PCGrad
        print(f"\n  [2/4] Training PCGrad (Gradient Surgery)...")
        cleanup_gpu()
        pcgrad_model = MonolithicMLP(pde.in_dim, pde.out_dim, hidden_dim=96, layers=4).to(device)
        t0 = time.time()
        hist_pcg = train_pcgrad_baseline(pde, pcgrad_model, epochs=ep, lr=1e-3, device=device)
        t_pcg = time.time() - t0
        tot_pcg, res_pcg, bc_pcg, _, rel_pcg = compute_model_losses(pcgrad_model, pde)
        entry["pcgrad_loss_total"] = tot_pcg; entry["pcgrad_loss_res"] = res_pcg; entry["pcgrad_loss_bc"] = bc_pcg; entry["pcgrad_rel_l2"] = rel_pcg; entry["pcgrad_time"] = t_pcg
        print(f"    PCGrad -> Total Loss: {tot_pcg:.6e} | Res: {res_pcg:.6e} | Rel L2: {rel_pcg:.4e} | Time: {t_pcg:.1f}s")
        
        # 3. CAGrad
        print(f"\n  [3/4] Training CAGrad (Conflict-Averse Gradient Descent)...")
        cleanup_gpu()
        cagrad_model = MonolithicMLP(pde.in_dim, pde.out_dim, hidden_dim=96, layers=4).to(device)
        t0 = time.time()
        hist_cag = train_cagrad_baseline(pde, cagrad_model, epochs=ep, lr=1e-3, c=0.5, device=device)
        t_cag = time.time() - t0
        tot_cag, res_cag, bc_cag, _, rel_cag = compute_model_losses(cagrad_model, pde)
        entry["cagrad_loss_total"] = tot_cag; entry["cagrad_loss_res"] = res_cag; entry["cagrad_loss_bc"] = bc_cag; entry["cagrad_rel_l2"] = rel_cag; entry["cagrad_time"] = t_cag
        print(f"    CAGrad -> Total Loss: {tot_cag:.6e} | Res: {res_cag:.6e} | Rel L2: {rel_cag:.4e} | Time: {t_cag:.1f}s")
        
        # 4. Proposed AS-PINN
        print(f"\n  [4/4] Training Proposed AS-PINN (Physics-Driven AMR Discovery + Production)...")
        cleanup_gpu()
        trainer = TwoStageASPINNTrainer(pde=pde, device=device)
        stage1_info = trainer.run_stage1_discovery(max_epochs=min(ep, 600))
        aspinn_model, summary_as = trainer.train_stage2_production(
            stage1_info, hidden_dim=64, layers=4, adamw_epochs=ep, lbfgs_steps=lbfgs_s
        )
        tot_as, res_as, bc_as, _, rel_as = compute_model_losses(aspinn_model, pde)
        t_as = stage1_info["stage1_time"] + summary_as["wall_time"]
        n_sub = summary_as["num_subspaces"]
        entry["aspinn_loss_total"] = tot_as; entry["aspinn_loss_res"] = res_as; entry["aspinn_loss_bc"] = bc_as; entry["aspinn_rel_l2"] = rel_as; entry["aspinn_time"] = t_as; entry["aspinn_subspaces"] = n_sub
        print(f"    Proposed AS-PINN (N*={n_sub}) -> Total Loss: {tot_as:.6e} | Res: {res_as:.6e} | Rel L2: {rel_as:.4e} | Time: {t_as:.1f}s")
        
        results.append(entry)
        subspace_summary_records[pde_key] = {
            "pde_name": pde_name,
            "stage1": stage1_info,
            "centroids": stage1_info["centroids"].clone().detach(),
            "num_subspaces": n_sub,
            "bandwidth": stage1_info["bandwidth"],
        }
        
        print(f"\n  >>> Rendering 4 Dedicated LaTeX-Style Figures for {pde_name}...")
        p1_res = os.path.join("results/plots", f"{pde_key}_convergence.png")
        plot_individual_convergence(pde_key, pde_name, hist_pinn, hist_pcg, hist_cag, summary_as["history"], p1_res)
        shutil.copy(p1_res, os.path.join("manuscript", f"{pde_key}_convergence.png"))
        
        p2_res = os.path.join("results/plots", f"{pde_key}_solutions.png")
        plot_solutions_comparison(pde_key, pde_name, pde, pinn_model, pcgrad_model, cagrad_model, aspinn_model, p2_res)
        shutil.copy(p2_res, os.path.join("manuscript", f"{pde_key}_solutions.png"))
        
        p3_res = os.path.join("results/plots", f"{pde_key}_territory.png")
        plot_subspace_territory_map(pde_key, pde_name, pde, aspinn_model, stage1_info, p3_res)
        shutil.copy(p3_res, os.path.join("manuscript", f"{pde_key}_territory.png"))
        
        p4_res = os.path.join("results/plots", f"{pde_key}_pou_profile.png")
        plot_pou_overlap_profile(pde_key, pde_name, pde, aspinn_model, stage1_info, p4_res)
        shutil.copy(p4_res, os.path.join("manuscript", f"{pde_key}_pou_profile.png"))
        
        del pinn_model, pcgrad_model, cagrad_model, aspinn_model, trainer
        cleanup_gpu()
        
        with open("results/data/master_benchmark_summary.json", "w") as f:
            json.dump(results, f, indent=2)
            
        keys = list(results[0].keys())
        with open("results/data/master_benchmark_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
            
    print("\n>>> Rendering Master 9-PDE Subspaces Grid Plot (3x3)...")
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes_flat = axes.ravel()
    
    idx = 0
    for pde_key, rec in subspace_summary_records.items():
        ax = axes_flat[idx]
        num_s = rec["num_subspaces"]
        c_np = rec["centroids"].cpu().numpy()
        sigma = rec["bandwidth"]
        nx, ny = 150, 150
        
        if pde_key == "burgers1d":
            b = ((-1, 1), (0, 1))
            X, Y = np.meshgrid(np.linspace(b[0][0], b[0][1], nx), np.linspace(b[1][0], b[1][1], ny))
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "convection2d":
            two_pi = float(2.0 * np.pi)
            X, Y = np.meshgrid(np.linspace(0.0, two_pi, nx), np.linspace(0.0, two_pi, ny))
            t_c = 0.50 * np.ones_like(X.ravel())
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "quantum_nls":
            X, Y = np.meshgrid(np.linspace(-3.0, 3.0, nx), np.linspace(-3.0, 3.0, ny))
            t_c = float(np.pi / 4.0) * np.ones_like(X.ravel())
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "helmholtz2d":
            X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "high_dim4d":
            X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
            zeros = np.zeros_like(X.ravel())
            t_c = 0.20 * np.ones_like(X.ravel())
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), zeros, zeros, t_c], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "bratu2d":
            X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "kovasznay":
            X, Y = np.meshgrid(np.linspace(-0.5, 1.0, nx), np.linspace(-0.5, 1.5, ny))
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "cavity_ns":
            X, Y = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        elif pde_key == "klein_gordon":
            X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
            t_c = 0.50 * np.ones_like(X.ravel())
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), t_c], axis=1), dtype=torch.float32, device=device)
            
        with torch.no_grad():
            c_t = rec["centroids"].to(device)
            diff = pts.unsqueeze(1) - c_t.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            logits = -dist_sq / (2.0 * (sigma ** 2) + 1e-8)
            psi = torch.softmax(logits, dim=-1).cpu().numpy()
            terr = np.argmax(psi, axis=1).reshape(ny, nx)
            
        cmap_g = plt.colormaps.get_cmap("tab10" if num_s <= 10 else "tab20")
        im = ax.contourf(X, Y, terr, levels=np.arange(num_s + 1) - 0.5, cmap=cmap_g)
        if c_np.shape[1] >= 2:
            ax.scatter(c_np[:, 0], c_np[:, 1], c="white", edgecolors="black", s=90, marker="o", linewidths=1.5)
        ax.set_title(f"{rec['pde_name']} ($N^*={num_s}$)", fontweight="bold", fontsize=12)
        idx += 1
        
    fig.suptitle("AS-PINN Autonomous Subspace Partition Architecture Across All 9 Benchmark PDEs", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()
    all_sub_path = "results/plots/all_subspaces_summary.png"
    fig.savefig(all_sub_path, bbox_inches="tight")
    fig.savefig("manuscript/all_subspaces_summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved Master 9-PDE Subspaces Grid: {all_sub_path}")
    
    print("\n" + "="*115)
    print("                              FINAL 4-WAY BENCHMARK SUMMARY (TOTAL LOSS L(Theta))")
    print("="*115)
    print(f"{'PDE Benchmark':<42} | {'Standard PINN':<18} | {'PCGrad':<18} | {'CAGrad':<18} | {'Proposed AS-PINN':<20}")
    print("-"*115)
    for r in results:
        pde_str = r['benchmark']
        pinn_str = f"{r['pinn_loss_total']:.4e}"
        pcg_str = f"{r['pcgrad_loss_total']:.4e}"
        cag_str = f"{r['cagrad_loss_total']:.4e}"
        as_str = f"{r['aspinn_loss_total']:.4e} (N*={r['aspinn_subspaces']})"
        print(f"{pde_str:<42} | {pinn_str:<18} | {pcg_str:<18} | {cag_str:<18} | {as_str:<20}")
    print("="*115)
    print("\n[+] Full results exported to results/data/master_benchmark_summary.json & .csv!")

if __name__ == "__main__":
    run_master_suite()
