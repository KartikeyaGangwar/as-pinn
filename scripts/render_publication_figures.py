"""
Publication-Grade Figure Generation Script for ATLAS (CMAME / JCP Standard).
Strictly adheres to Nicolas P. Rougier's 'Scientific Visualization: Python + Matplotlib'
(Ten Simple Rules, Typography, Layout, and Chartjunk Elimination).

Features:
1. Native LaTeX Typography (Computer Modern Roman via MiKTeX / tex.usetex = True).
2. Pure Decoupled Figures:
   - Figure 1: 2x2 Hero Subspaces Summary (Burgers, Convection, Helmholtz, Cavity)
   - Figure 2: 2x2 Hero Convergence Trajectories (Log Loss Scale)
   - Figure 7 (Appendix A): 3+2 Centered Mosaic Supplementary Subspaces
   - Figure 8 (Appendix A): 3+2 Centered Mosaic Supplementary Convergence
3. Zero Chartjunk:
   - Eliminates all in-image presentation titles and double headers.
   - Eliminates all ghost text bleeding and chopped exponents.
   - Strips top suptitles from solution comparison plots so only clean subpanels remain.
"""

import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configure Matplotlib for exact Nicolas Rougier / CMAME LaTeX Typography
# Gracefully falls back to bundled Mathtext Computer Modern if LaTeX is absent
# ---------------------------------------------------------------------------
has_latex = shutil.which("latex") is not None
if has_latex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
else:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

def clean_title(title_str):
    if has_latex:
        return title_str
    # Strip TeX macro wrappers for standard Matplotlib rendering
    t = title_str.replace(r"\textbf{", "").replace("}", "")
    t = t.replace(r"\'e", "é").replace(r'\"o', "ö")
    return t


def extract_clean_plot_body(img_path, gap_y=100, pad=10):
    """
    Extracts the clean axes frame, ticks, data, and legend from a standalone plot,
    stripping away redundant suptitles or in-image presentation headers.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing plot file: {img_path}")
        
    im = Image.open(img_path)
    # The title text ends before gap_y; slice from gap_y downwards
    cropped = im.crop((0, gap_y, im.width, im.height))
    
    # Auto-trim outer white borders
    c_gray = np.array(cropped.convert('L'))
    nonwhite = np.where(c_gray < 254)
    if len(nonwhite[0]) == 0:
        return cropped
        
    ymin, ymax = nonwhite[0].min(), nonwhite[0].max()
    xmin, xmax = nonwhite[1].min(), nonwhite[1].max()
    
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(cropped.height, ymax + pad)
    xmax = min(cropped.width, xmax + pad)
    return cropped.crop((xmin, ymin, xmax, ymax))


def render_all_publication_composites():
    print("[*] Starting Publication-Grade Composite Rendering (Rougier Standards)...")
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("manuscript", exist_ok=True)
    
    # Hero benchmarks configuration
    hero_specs = [
        ("burgers1d", r"\textbf{(a) 1D Viscous Burgers' Shock} ($N^*=7$)", r"\textbf{(a) 1D Viscous Burgers' Equation}"),
        ("convection2d", r"\textbf{(b) 2D High-P\'eclet Convection} ($N^*=22$)", r"\textbf{(b) 2D High-P\'eclet Convection Transport}"),
        ("helmholtz2d", r"\textbf{(c) 2D High-Frequency Helmholtz} ($N^*=21$)", r"\textbf{(c) 2D High-Frequency Helmholtz Equation}"),
        ("cavity_ns", r"\textbf{(d) 2D Lid-Driven Cavity Navier-Stokes} ($N^*=22$)", r"\textbf{(d) 2D Lid-Driven Cavity Navier-Stokes}"),
    ]
    
    # Supplementary benchmarks configuration
    supp_specs = [
        ("quantum_nls", r"\textbf{(a) 2D Quantum NLS Soliton} ($N^*=21$)", r"\textbf{(a) 2D Quantum Non-Linear Schr\"odinger}"),
        ("high_dim4d", r"\textbf{(b) 5D Space-Time Diffusion} ($N^*=24$)", r"\textbf{(b) 5D Space-Time Isotropic Diffusion}"),
        ("bratu2d", r"\textbf{(c) 2D Non-Linear Bratu Ignition} ($N^*=17$)", r"\textbf{(c) 2D Non-Linear Bratu Thermal Ignition}"),
        ("kovasznay", r"\textbf{(d) 2D Steady Kovasznay Flow} ($N^*=7$)", r"\textbf{(d) 2D Steady Kovasznay Flow}"),
        ("klein_gordon", r"\textbf{(e) 2D Relativistic Klein-Gordon} ($N^*=21$)", r"\textbf{(e) 2D Relativistic Klein-Gordon Wave}"),
    ]
    
    # -----------------------------------------------------------------------
    # 1. FIGURE 1: HERO SUBSPACES SUMMARY (2x2 Grid)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    for idx, (pde_key, s_title, _) in enumerate(hero_specs):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        img_path = f"results/plots/{pde_key}_territory.png"
        clean_img = extract_clean_plot_body(img_path, gap_y=95)
        ax.imshow(clean_img)
        ax.axis("off")
        ax.set_title(clean_title(s_title), fontsize=14, pad=12, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig("manuscript/hero_subspaces_summary.png", bbox_inches="tight", dpi=300)
    fig.savefig("results/plots/hero_subspaces_summary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [+] Rendered Clean Figure 1: hero_subspaces_summary.png")
    
    # -----------------------------------------------------------------------
    # 2. FIGURE 2: HERO CONVERGENCE SUMMARY (2x2 Grid)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    for idx, (pde_key, _, c_title) in enumerate(hero_specs):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        img_path = f"results/plots/{pde_key}_convergence.png"
        clean_img = extract_clean_plot_body(img_path, gap_y=95)
        ax.imshow(clean_img)
        ax.axis("off")
        ax.set_title(clean_title(c_title), fontsize=14, pad=12, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig("manuscript/hero_convergence_summary.png", bbox_inches="tight", dpi=300)
    fig.savefig("results/plots/hero_convergence_summary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [+] Rendered Clean Figure 2: hero_convergence_summary.png")
    
    # -----------------------------------------------------------------------
    # 3. FIGURE 7 (APPENDIX A): SUPPLEMENTARY SUBSPACES (3+2 Mosaic)
    # -----------------------------------------------------------------------
    mosaic = [
        ['s0', 's0', 's1', 's1', 's2', 's2'],
        ['.', 's3', 's3', 's4', 's4', '.']
    ]
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(18, 12), dpi=300)
    for idx, (pde_key, s_title, _) in enumerate(supp_specs):
        ax = ax_dict[f's{idx}']
        img_path = f"results/plots/{pde_key}_territory.png"
        clean_img = extract_clean_plot_body(img_path, gap_y=95)
        ax.imshow(clean_img)
        ax.axis("off")
        ax.set_title(clean_title(s_title), fontsize=13, pad=10, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig("manuscript/supp_subspaces_summary.png", bbox_inches="tight", dpi=300)
    fig.savefig("results/plots/supp_subspaces_summary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [+] Rendered Clean Figure 7: supp_subspaces_summary.png")
    
    # -----------------------------------------------------------------------
    # 4. FIGURE 8 (APPENDIX A): SUPPLEMENTARY CONVERGENCE (3+2 Mosaic)
    # -----------------------------------------------------------------------
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(18, 12), dpi=300)
    for idx, (pde_key, _, c_title) in enumerate(supp_specs):
        ax = ax_dict[f's{idx}']
        img_path = f"results/plots/{pde_key}_convergence.png"
        clean_img = extract_clean_plot_body(img_path, gap_y=95)
        ax.imshow(clean_img)
        ax.axis("off")
        ax.set_title(clean_title(c_title), fontsize=13, pad=10, fontweight="bold")

        
    plt.tight_layout()
    fig.savefig("manuscript/supp_convergence_summary.png", bbox_inches="tight", dpi=300)
    fig.savefig("results/plots/supp_convergence_summary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [+] Rendered Clean Figure 8: supp_convergence_summary.png")

    # -----------------------------------------------------------------------
    # 5. CLEAN SOLUTION COMPARISON FIGURES (Strip Presentation Suptitles)
    # -----------------------------------------------------------------------
    all_pdes = [p[0] for p in hero_specs] + [p[0] for p in supp_specs]
    for pde_key in all_pdes:
        sol_path = f"results/plots/{pde_key}_solutions.png"
        if os.path.exists(sol_path):
            im = Image.open(sol_path)
            # The suptitle sits in y=20..95, gap is around 100..140
            # Slicing from y=120 leaves subpanels intact and removes the suptitle
            clean_sol = extract_clean_plot_body(sol_path, gap_y=120, pad=5)
            clean_sol.save(f"manuscript/{pde_key}_solutions.png")
            clean_sol.save(f"results/plots/{pde_key}_solutions.png")
            print(f"  [+] Cleaned solutions plot: {pde_key}_solutions.png")

    print("[*] All publication figures rendered cleanly with zero chartjunk.")

if __name__ == "__main__":
    render_all_publication_composites()
