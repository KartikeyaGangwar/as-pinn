"""
Post-processing and visualization script for lid-driven cavity flow solutions.

Loads precomputed flow fields from a NumPy .npz file and generates
high-quality showcase plots suitable for GitHub README or reports.

Usage:
    python plot_from_npz.py --file flow_fields_Re1000_N251.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import io
import argparse

# UNICODE FIX FOR TERMINALS
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# ---------------------------------------------------------------------
# Global plotting style (bold, clean, confident)
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})


def get_vortex_center(psi, x, y):
    """Return physical coordinates of primary vortex center."""
    idx = np.unravel_index(np.argmin(psi), psi.shape)
    return x[idx[1]], y[idx[0]]


def create_showcase_plots(data, filename, save=True):
    """Create and display showcase plots for lid-driven cavity flow."""
    # -----------------------------
    # Extract data
    # -----------------------------
    x, y = data["x"], data["y"]
    X, Y = data["X"], data["Y"]
    psi, u, v = data["psi"], data["u"], data["v"]
    L = x[-1]

    # Reynolds number (from filename fallback)
    m = re.search(r"Re(\d+)", filename)
    Re = m.group(1) if m else "?"

    # Vortex center
    idx = np.unravel_index(np.argmin(psi), psi.shape)
    vortex_x, vortex_y = x[idx[1]], y[idx[0]]

    # -----------------------------
    # Plot style
    # -----------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axs = plt.subplots(
        2, 2,
        figsize=(16, 14),
        dpi=300
    )

    # ==================================================
    # (1,1) STREAMLINES
    # ==================================================
    ax = axs[0, 0]
    levels = np.linspace(psi.min(), 0.0, 30)

    ax.contourf(X, Y, psi, levels=levels, cmap="viridis", alpha=0.75)
    ax.contour(X, Y, psi, levels=levels, colors="k", linewidths=0.6)

    ax.set_title("Streamlines (ψ)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.grid(False)

    # ==================================================
    # (1,2) VELOCITY MAGNITUDE
    # ==================================================
    ax = axs[0, 1]
    vel_mag = np.sqrt(u**2 + v**2)

    cf = ax.contourf(X, Y, vel_mag, levels=30, cmap="plasma")

    ax.set_title("Velocity Magnitude |V|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # ==================================================
    # (2,1) u-VELOCITY CENTERLINE
    # ==================================================
    ax = axs[1, 0]
    x_mid = np.argmin(np.abs(x - L / 2))
    ax.plot(u[:, x_mid], y, lw=3)

    ax.set_title("u-velocity at x = L/2")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.4)

    # ==================================================
    # (2,2) v-VELOCITY CENTERLINE
    # ==================================================
    ax = axs[1, 1]
    y_mid = np.argmin(np.abs(y - L / 2))
    ax.plot(x, v[y_mid, :], lw=3, color="tab:red")

    ax.set_title("v-velocity at y = L/2")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.grid(True, ls="--", alpha=0.4)

    # ==================================================
    # SUPERTITLE
    # ==================================================
    fig.suptitle(
        f"Lid-Driven Cavity Flow (Re = {Re})",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out = f"lid_driven_cavity_showcase_Re{Re}.png"
        plt.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"[SUCCESS] Saved: {out}")

    plt.show()

# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # --- Detect notebook / Colab ---
    running_in_notebook = "ipykernel" in sys.modules

    if running_in_notebook:
        # DEFAULT FILE FOR NOTEBOOK USE
        file_path = "flow_fields_Re1000_N251.npz"
        print(f"[INFO] Notebook detected. Using default file: {file_path}")
    else:
        parser = argparse.ArgumentParser(
            description="Plot lid-driven cavity solution from .npz file"
        )
        parser.add_argument(
            "--file", type=str, required=True,
            help="Path to .npz file containing saved flow fields"
        )
        args = parser.parse_args()
        file_path = args.file

    data = np.load(file_path, allow_pickle=True)
    create_showcase_plots(data, filename=file_path)
