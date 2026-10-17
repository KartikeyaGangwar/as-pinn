"""
ATLAS Multi-GPU Parallel Benchmark Suite Runner.
Enables concurrent execution across multiple GPUs (e.g., Kaggle GPU 2x T4).

Workload Partitioning:
- GPU 0 (cuda:0): burgers1d, convection2d, helmholtz2d, cavity_ns, kovasznay  (~8,000 epochs)
- GPU 1 (cuda:1): quantum_nls, high_dim4d, bratu2d, klein_gordon             (~7,000 epochs)

Both workers execute simultaneously in parallel CUDA streams, cutting runtime by ~50%.
Gracefully falls back to sequential single-device execution if < 2 GPUs are detected.
"""

import os
import sys
import time
import json
import csv
import glob
import threading
import subprocess
import torch

from scripts.render_publication_figures import render_all_publication_composites


def stream_reader(pipe, prefix):
    try:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
    except Exception:
        pass
    finally:
        pipe.close()


def merge_and_print_summary():
    print("\n" + "="*115)
    print("                      CONSOLIDATING MULTI-GPU BENCHMARK SUMMARY (ALL 9 PDES)")
    print("="*115)
    
    all_keys = [
        "burgers1d", "convection2d", "quantum_nls", "helmholtz2d", 
        "high_dim4d", "bratu2d", "kovasznay", "cavity_ns", "klein_gordon"
    ]
    
    consolidated = []
    for k in all_keys:
        p_path = f"results/data/{k}_summary.json"
        if os.path.exists(p_path):
            with open(p_path, "r") as f:
                consolidated.append(json.load(f))
        else:
            print(f"[!] Warning: Summary for '{k}' not found at {p_path}")
            
    if consolidated:
        with open("results/data/master_benchmark_summary.json", "w") as f:
            json.dump(consolidated, f, indent=2)
            
        keys = list(consolidated[0].keys())
        with open("results/data/master_benchmark_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(consolidated)
            
        print(f"{'PDE Benchmark':<42} | {'Standard PINN':<18} | {'PCGrad':<18} | {'CAGrad':<18} | {'Proposed ATLAS':<20}")
        print("-"*115)
        for r in consolidated:
            pde_str = r.get('benchmark', 'Unknown')
            pinn_str = f"{r.get('pinn_loss_total', 0.0):.4e}"
            pcg_str = f"{r.get('pcgrad_loss_total', 0.0):.4e}"
            cag_str = f"{r.get('cagrad_loss_total', 0.0):.4e}"
            as_str = f"{r.get('aspinn_loss_total', 0.0):.4e} (N*={r.get('aspinn_subspaces', '?')})"
            print(f"{pde_str:<42} | {pinn_str:<18} | {pcg_str:<18} | {cag_str:<18} | {as_str:<20}")
        print("="*115)
        print(f"[+] Consolidated summary saved to results/data/master_benchmark_summary.json & .csv")


def run_parallel_suite():
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("="*115)
    print("         ATLAS MULTI-GPU PARALLEL ACCELERATOR (PYTORCH)")
    print("="*115)
    print(f"[*] Detected CUDA Devices: {num_gpus}")
    
    start_time = time.time()
    
    if num_gpus >= 2:
        print(f"[+] Multi-GPU Acceleration Active:")
        print(f"    - Device 0: {torch.cuda.get_device_name(0)} (Assigned Worker 0)")
        print(f"    - Device 1: {torch.cuda.get_device_name(1)} (Assigned Worker 1)")
        print("="*115)
        
        # Balanced PDE partitioning (8,000 epochs vs 7,000 epochs)
        pdes_gpu0 = "burgers1d,convection2d,helmholtz2d,cavity_ns,kovasznay"
        pdes_gpu1 = "quantum_nls,high_dim4d,bratu2d,klein_gordon"
        
        print(f"[*] Worker 0 [GPU 0] executing: {pdes_gpu0}")
        print(f"[*] Worker 1 [GPU 1] executing: {pdes_gpu1}")
        print("="*115)
        
        cmd0 = [
            sys.executable, "-u", "benchmarks/run_master_benchmarks.py",
            "--device", "cuda:0",
            "--pdes", pdes_gpu0,
            "--skip-summary-grid"
        ]
        cmd1 = [
            sys.executable, "-u", "benchmarks/run_master_benchmarks.py",
            "--device", "cuda:1",
            "--pdes", pdes_gpu1,
            "--skip-summary-grid"
        ]
        
        p0 = subprocess.Popen(cmd0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        t0 = threading.Thread(target=stream_reader, args=(p0.stdout, "GPU 0"))
        t1 = threading.Thread(target=stream_reader, args=(p1.stdout, "GPU 1"))
        t0.daemon = True
        t1.daemon = True
        t0.start()
        t1.start()
        
        ret0 = p0.wait()
        ret1 = p1.wait()
        t0.join()
        t1.join()
        
        if ret0 != 0 or ret1 != 0:
            print(f"[!] ERROR: Worker process failure! Exit codes: GPU 0 = {ret0}, GPU 1 = {ret1}")
            sys.exit(1)
            
        print("\n[+] Both GPU workers completed execution successfully!")
        
    else:
        dev_name = torch.cuda.get_device_name(0) if num_gpus == 1 else "CPU"
        print(f"[*] Single compute device detected: {dev_name}")
        print(f"[*] Executing full 9-PDE benchmark suite sequentially...")
        print("="*115)
        
        cmd = [sys.executable, "-u", "benchmarks/run_master_benchmarks.py"]
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[!] Benchmark execution returned non-zero exit code: {ret}")
            sys.exit(ret)
            
    # Post-process: Consolidate summary table & render publication figures
    merge_and_print_summary()
    
    print("\n[*] Rendering Rougier-Standard Publication Composites...")
    render_all_publication_composites()
    
    elapsed = time.time() - start_time
    print("\n" + "="*115)
    print(f"[+] ALL BENCHMARKS, CHECKPOINTS, AND PUBLICATION FIGURES READY in {elapsed/60:.2f} minutes!")
    print("="*115)


if __name__ == "__main__":
    run_parallel_suite()
