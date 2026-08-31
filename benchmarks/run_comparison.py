import argparse
import json
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.as_pinn import AdaptiveSubspacePINN
from physics.burgers import Burgers1D
from physics.elliptic_interface import EllipticInterface2D
from physics.helmholtz import Helmholtz2D
from physics.allen_cahn import AllenCahn1D
from physics.high_dim_diffusion import HighDimDiffusion4D
from physics.cshape_magnet import CShapeElectromagnet
from physics.meissner_cylinder import MeissnerCylinder
from train_two_stage_as_pinn import TwoStageASPINNTrainer
from benchmarks.models_baseline import (
    MonolithicMLP,
    StaticFBPINN,
    train_model_adamw_lbfgs,
    train_pcgrad_baseline,
    train_cagrad_baseline,
)


def instantiate_pde(pde_name: str, device: torch.device):
    if pde_name == "burgers1d":
        pde = Burgers1D(nu=0.01/np.pi, device=device)
        bounds = ((-1.0, 1.0), (0.0, 1.0))
        grid_shape = (2, 2)
    elif pde_name == "elliptic2d":
        pde = EllipticInterface2D(a1=1.0, a2=1000.0, r0=0.5, device=device)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        grid_shape = (2, 2)
    elif pde_name == "helmholtz2d":
        pde = Helmholtz2D(k=4.0*np.pi, device=device)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        grid_shape = (2, 2)
    elif pde_name == "allen_cahn":
        pde = AllenCahn1D(epsilon=0.02, velocity=0.5, reaction_coeff=5.0, device=device)
        bounds = ((-1.0, 1.0), (0.0, 1.0))
        grid_shape = (2, 2)
    elif pde_name == "high_dim4d":
        pde = HighDimDiffusion4D(kappa=0.1, alpha=0.5, device=device)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))
        grid_shape = (2, 2, 2, 2, 2)
    elif pde_name == "cshape":
        pde = CShapeElectromagnet(device=device)
        bounds = ((-8.0, 8.0), (-8.0, 8.0))
        grid_shape = (2, 2)
    elif pde_name == "meissner":
        pde = MeissnerCylinder(r0=1.0, r_outer=4.0, b0=1.0, device=device)
        bounds = ((-4.0, 4.0), (-4.0, 4.0))
        grid_shape = (2, 2)
    else:
        raise ValueError(f"Unknown PDE: {pde_name}")
    pde.bounds = bounds
    return pde, bounds, grid_shape


def run_benchmark_for_pde(
    pde_name: str,
    adamw_epochs: int = 1000,
    lbfgs_steps: int = 150,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "#"*75)
    print(f"  EXECUTING FULL BENCHMARK: {pde_name.upper()} on {device}")
    print(f"  AdamW Epochs: {adamw_epochs} | L-BFGS Steps: {lbfgs_steps}")
    print("#"*75)
    
    pde, bounds, grid_shape = instantiate_pde(pde_name, device)
    results = {}
    
    # 1. Monolithic PINN (AdamW + Combined L-BFGS)
    print("\n--- [1/6] Monolithic PINN (AdamW + L-BFGS) ---")
    mono_model = MonolithicMLP(pde.in_dim, pde.out_dim, hidden_dim=96, layers=4, activation="tanh").to(device)
    n_params_mono = sum(p.numel() for p in mono_model.parameters() if p.requires_grad)
    mono_hist = train_model_adamw_lbfgs(pde, mono_model, adamw_epochs=adamw_epochs, lbfgs_steps=lbfgs_steps, device=device)
    results["Monolithic PINN"] = {
        "history": mono_hist,
        "final_rel_l2": mono_hist["rel_l2_err"][-1],
        "final_linf": mono_hist["linf_err"][-1],
        "total_params": n_params_mono,
        "wall_time": mono_hist["wall_time"][-1],
    }
    print(f"  [+] Monolithic Final Rel L2: {results['Monolithic PINN']['final_rel_l2']:.4e} in {results['Monolithic PINN']['wall_time']:.1f}s")
    
    # 2. Monolithic + PCGrad
    print("\n--- [2/6] Monolithic + PCGrad (AdamW 1000) ---")
    pcg_model = MonolithicMLP(pde.in_dim, pde.out_dim, hidden_dim=96, layers=4, activation="tanh").to(device)
    n_params_pcg = sum(p.numel() for p in pcg_model.parameters() if p.requires_grad)
    pcg_hist = train_pcgrad_baseline(pde, pcg_model, epochs=adamw_epochs, device=device)
    results["Monolithic + PCGrad"] = {
        "history": pcg_hist,
        "final_rel_l2": pcg_hist["rel_l2_err"][-1],
        "final_linf": pcg_hist["linf_err"][-1],
        "total_params": n_params_pcg,
        "wall_time": pcg_hist["wall_time"][-1],
    }
    print(f"  [+] PCGrad Final Rel L2: {results['Monolithic + PCGrad']['final_rel_l2']:.4e} in {results['Monolithic + PCGrad']['wall_time']:.1f}s")
    
    # 3. Monolithic + CAGrad
    print("\n--- [3/6] Monolithic + CAGrad (AdamW 1000) ---")
    cag_model = MonolithicMLP(pde.in_dim, pde.out_dim, hidden_dim=96, layers=4, activation="tanh").to(device)
    n_params_cag = sum(p.numel() for p in cag_model.parameters() if p.requires_grad)
    cag_hist = train_cagrad_baseline(pde, cag_model, epochs=adamw_epochs, c=0.5, device=device)
    results["Monolithic + CAGrad"] = {
        "history": cag_hist,
        "final_rel_l2": cag_hist["rel_l2_err"][-1],
        "final_linf": cag_hist["linf_err"][-1],
        "total_params": n_params_cag,
        "wall_time": cag_hist["wall_time"][-1],
    }
    print(f"  [+] CAGrad Final Rel L2: {results['Monolithic + CAGrad']['final_rel_l2']:.4e} in {results['Monolithic + CAGrad']['wall_time']:.1f}s")
    
    # 4. Static FBPINN (Uniform N=4 Grid + L-BFGS)
    print("\n--- [4/6] Static FBPINN (Uniform N=4 + L-BFGS) ---")
    fb_model = StaticFBPINN(pde.in_dim, pde.out_dim, grid_shape=grid_shape, bounds=bounds, hidden_dim=96, layers=4, activation="tanh").to(device)
    n_params_fb = sum(p.numel() for p in fb_model.parameters() if p.requires_grad)
    fb_hist = train_model_adamw_lbfgs(pde, fb_model, adamw_epochs=adamw_epochs, lbfgs_steps=lbfgs_steps, device=device)
    results["Static FBPINN (N=4)"] = {
        "history": fb_hist,
        "final_rel_l2": fb_hist["rel_l2_err"][-1],
        "final_linf": fb_hist["linf_err"][-1],
        "total_params": n_params_fb,
        "wall_time": fb_hist["wall_time"][-1],
    }
    print(f"  [+] Static FBPINN Final Rel L2: {results['Static FBPINN (N=4)']['final_rel_l2']:.4e} in {results['Static FBPINN (N=4)']['wall_time']:.1f}s")
    
    # 5. Two-Stage AS-PINN Engine
    print("\n--- [5/6] Stage 1 Discovery & Two-Stage AS-PINN (Equal Capacity) ---")
    trainer = TwoStageASPINNTrainer(pde=pde, device=device)
    stage1_info = trainer.run_stage1_discovery(
        min_epochs=300,
        max_epochs=500,
        warmup_epochs=60,
        profile_freq=25,
        cooldown_epochs=25,
        conflict_threshold=0.15,
        max_subspaces=5,
    )
    
    m_equal, res_equal = trainer.train_stage2_production(
        discovered_info=stage1_info,
        hidden_dim=96,
        layers=4,
        adamw_epochs=adamw_epochs,
        lbfgs_steps=lbfgs_steps,
    )
    results["AS-PINN (Equal Capacity)"] = {
        "history": res_equal["history"],
        "final_rel_l2": res_equal["final_rel_l2"],
        "final_linf": res_equal["final_linf"],
        "total_params": res_equal["total_params"],
        "wall_time": res_equal["wall_time"] + stage1_info["discovery_time"],
        "final_subspaces": res_equal["num_subspaces"],
    }
    print(f"  [+] AS-PINN (Equal Capacity) Final Rel L2: {results['AS-PINN (Equal Capacity)']['final_rel_l2']:.4e}")
    
    # 6. Two-Stage AS-PINN (Iso-Parameter Budget)
    print("\n--- [6/6] Two-Stage AS-PINN (Iso-Parameter Budget) ---")
    m_iso, res_iso = trainer.train_stage2_production(
        discovered_info=stage1_info,
        hidden_dim=44,
        layers=3,
        adamw_epochs=adamw_epochs,
        lbfgs_steps=lbfgs_steps,
    )
    results["AS-PINN (Iso-Param Budget)"] = {
        "history": res_iso["history"],
        "final_rel_l2": res_iso["final_rel_l2"],
        "final_linf": res_iso["final_linf"],
        "total_params": res_iso["total_params"],
        "wall_time": res_iso["wall_time"] + stage1_info["discovery_time"],
        "final_subspaces": res_iso["num_subspaces"],
    }
    print(f"  [+] AS-PINN (Iso-Param Budget) Final Rel L2: {results['AS-PINN (Iso-Param Budget)']['final_rel_l2']:.4e}")
    
    # Save clean summary JSON
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    summary_path = f"results/data/{pde_name}_summary.json"
    clean_summary = {
        k: {
            "final_rel_l2": v["final_rel_l2"],
            "final_linf": v["final_linf"],
            "total_params": v["total_params"],
            "wall_time": v["wall_time"],
            "final_subspaces": v.get("final_subspaces", 1),
        }
        for k, v in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(clean_summary, f, indent=2)
        
    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "Monolithic PINN": "crimson",
        "Monolithic + PCGrad": "darkorange",
        "Monolithic + CAGrad": "purple",
        "Static FBPINN (N=4)": "royalblue",
        "AS-PINN (Equal Capacity)": "forestgreen",
        "AS-PINN (Iso-Param Budget)": "teal",
    }
    styles = {
        "Monolithic PINN": "--",
        "Monolithic + PCGrad": "-.",
        "Monolithic + CAGrad": ":",
        "Static FBPINN (N=4)": "--",
        "AS-PINN (Equal Capacity)": "-",
        "AS-PINN (Iso-Param Budget)": "-.",
    }
    
    for name, res in results.items():
        hist = res["history"]
        axes[0].semilogy(hist["epoch"], hist["rel_l2_err"], label=f"{name} ({res['final_rel_l2']:.2e})", color=colors.get(name, "black"), linestyle=styles.get(name, "-"), linewidth=2.2)
        axes[1].semilogy(hist["wall_time"], hist["rel_l2_err"], label=name, color=colors.get(name, "black"), linestyle=styles.get(name, "-"), linewidth=2.2)

    axes[0].set_xlabel("Epochs (AdamW + L-BFGS)", fontsize=12)
    axes[0].set_ylabel("Relative $L_2$ Error", fontsize=12)
    axes[0].set_title(f"Convergence vs Iterations ({pde_name.upper()})", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].set_xlabel("Wall-Clock Time (seconds)", fontsize=12)
    axes[1].set_ylabel("Relative $L_2$ Error", fontsize=12)
    axes[1].set_title(f"Convergence vs Time ({pde_name.upper()})", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plot_path = f"results/plots/{pde_name}_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Generate LaTeX Summary Table
    tex_path = f"results/tables/{pde_name}_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{Performance comparison on {pde_name.upper()} benchmark.}}\n")
        f.write("\\label{tab:" + pde_name + "_results}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Rel. $L_2$ Error} & \\textbf{$L_\\infty$ Error} & \\textbf{Params} & \\textbf{Time (s)} \\\\ \\hline\n")
        for name, row in clean_summary.items():
            f.write(f"{name} & {row['final_rel_l2']:.2e} & {row['final_linf']:.2e} & {row['total_params']} & {row['wall_time']:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        
    print(f"\n[+] {pde_name.upper()} complete. Summary saved to {summary_path}, {plot_path}, {tex_path}")
    return clean_summary


def run_all_benchmarks(adamw_epochs: int = 1000, lbfgs_steps: int = 150):
    all_pdes = [
        "burgers1d",
        "elliptic2d",
        "helmholtz2d",
        "allen_cahn",
        "cshape",
        "meissner",
        "high_dim4d",
    ]
    master_records = {}
    for pde_name in all_pdes:
        rec = run_benchmark_for_pde(pde_name, adamw_epochs=adamw_epochs, lbfgs_steps=lbfgs_steps)
        master_records[pde_name] = rec
        
    with open("results/data/master_7_pdes_summary.json", "w") as f:
        json.dump(master_records, f, indent=2)
    print("\n" + "="*80)
    print("  ALL 7 PDES BENCHMARK COMPLETE! MASTER SUMMARY SAVED.")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", type=str, default="all")
    parser.add_argument("--adamw_epochs", type=int, default=1000)
    parser.add_argument("--lbfgs_steps", type=int, default=150)
    args = parser.parse_args()
    
    if args.pde == "all":
        run_all_benchmarks(adamw_epochs=args.adamw_epochs, lbfgs_steps=args.lbfgs_steps)
    else:
        run_benchmark_for_pde(args.pde, adamw_epochs=args.adamw_epochs, lbfgs_steps=args.lbfgs_steps)
