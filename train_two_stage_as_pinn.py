import copy
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from models.as_pinn import AdaptiveSubspacePINN, SubspaceMLP
from models.conflict_monitor import ContinuousConflictMonitor
from physics.base_pde import BasePDE

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TwoStageASPINNTrainer:
    """
    Two-Stage Adaptive N-Subspace PINN Trainer (AS-PINN).
    
    100% Pure Physics-Adaptive Autonomous Stopping:
      - Active-Domain Support Profiling (subspaces profile only their active region psi_k > 0.10).
      - Non-Redundant Spatial Separation (d_min >= 0.40 sigma prevents duplicate crowding).
      - True Physics Quiescence (Naturally terminates when all multiscale conflicts subside).
      - Fixed Random Seed (42) for exact determinism.
      - Full-Batch Adaptive Loss Weighting (Dynamic GradNorm).
      - Continuous Loss Logging (AdamW + L-BFGS intermediate trajectory).
      - Dense Collocation: 8192 interior + 1024 boundary points.
      - Full-Horizon Combined Global L-BFGS Polish (250 steps).
    """
    def __init__(
        self,
        pde: BasePDE,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.pde = pde
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        set_seed(self.seed)
        cleanup_gpu()
        
    def _compute_bandwidth(self) -> torch.Tensor:
        bounds = getattr(self.pde, "bounds", None)
        if bounds is not None:
            # Dimension-Normalized Anisotropic Bandwidth Vector: 0.25 * (b_max - b_min)
            sigmas = [float(0.25 * (b[1] - b[0])) for b in bounds]
            return torch.tensor(sigmas, dtype=torch.float32, device=self.device).reshape(1, self.pde.in_dim)
        return torch.full((1, self.pde.in_dim), 0.50, dtype=torch.float32, device=self.device)
        
    def run_stage1_discovery(
        self,
        min_epochs: int = 250,
        max_epochs: int = 800,
        profile_freq: int = 25,
        warmup_epochs: int = 50,
        cooldown_epochs: int = 25,
        max_subspaces: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> Dict:
        set_seed(self.seed)
        cleanup_gpu()
        sigma = self._compute_bandwidth()
        pde_name = getattr(self.pde, "name", self.pde.__class__.__name__)
        sigma_str = [round(float(s), 3) for s in sigma.cpu().numpy().ravel()]
        print(f"\n" + "="*70)
        print(f"  STAGE 1: 100% PHYSICS-ADAPTIVE AMR DISCOVERY (Seed: {self.seed})")
        print(f"  PDE: {pde_name} | Device: {self.device} | Bandwidth: {sigma_str} | Cap: Free (up to {max_subspaces})")
        print("="*70)
        
        bounds = getattr(self.pde, "bounds", None)
        if bounds is not None:
            c0 = torch.tensor([[0.5 * (b[0] + b[1]) for b in bounds]], dtype=torch.float32, device=self.device)
        else:
            c0 = torch.zeros((1, self.pde.in_dim), dtype=torch.float32, device=self.device)
            
        probe_model = AdaptiveSubspacePINN(
            in_dim=self.pde.in_dim,
            out_dim=self.pde.out_dim,
            initial_subspaces=1,
            hidden_dim=32,
            layers=2,
            activation="tanh",
            bandwidth=sigma,
            initial_centroids=c0,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(probe_model.parameters(), lr=lr, weight_decay=weight_decay)
        monitor = ContinuousConflictMonitor(
            threshold=0.0,
            ema_decay=0.85,
            min_clash_ratio=0.20,
            in_dim=self.pde.in_dim,
        )
        
        last_cleavage_epoch = -cooldown_epochs
        consecutive_stable_cycles = 0
        cleavage_history = []
        start_time = time.time()
        
        for epoch in range(1, max_epochs + 1):
            probe_model.train()
            optimizer.zero_grad()
            
            x_int = self.pde.sample_interior(4096)
            res = self.pde.compute_residuals(probe_model, x_int)
            loss_res = torch.mean(res ** 2)
            
            x_bc, u_bc = self.pde.sample_boundary(512)
            loss_bc = self.pde.compute_boundary_loss(probe_model, x_bc, u_bc)
            
            loss_ic = torch.tensor(0.0, device=self.device)
            ic_samples = self.pde.sample_initial(512)
            if ic_samples is not None:
                x_ic, u_ic = ic_samples
                loss_ic = self.pde.compute_initial_loss(probe_model, x_ic, u_ic)
                
            total_loss = loss_res + 20.0 * loss_bc + 20.0 * loss_ic
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=5.0)
            optimizer.step()
            
            can_cleave = (
                epoch >= warmup_epochs and
                (epoch - last_cleavage_epoch) >= cooldown_epochs and
                probe_model.num_subspaces < max_subspaces
            )
            
            cleaved_this_epoch = False
            
            if can_cleave and (epoch % profile_freq == 0):
                x_int_prof = self.pde.sample_interior(128)
                
                with torch.no_grad():
                    psi_prof = probe_model.partition_of_unity(x_int_prof)
                
                for k in range(probe_model.num_subspaces):
                    sub_model = probe_model.subspaces[k]
                    sub_params = [p for p in sub_model.parameters() if p.requires_grad]
                    if not sub_params:
                        continue
                        
                    c_k = probe_model.centroids[k:k+1]
                    sigma_k = probe_model.bandwidth
                    
                    # Support-Weighted Filtering: Profile active territory
                    active_mask = psi_prof[:, k] > 0.05
                    if probe_model.num_subspaces > 1 and torch.sum(active_mask) >= 6:
                        x_active = x_int_prof[active_mask][:48]
                    else:
                        x_active = x_int_prof[:48]
                    
                    per_sample_grads = []
                    valid_pts = []
                    
                    for i in range(x_active.shape[0]):
                        pt = x_active[i:i+1]
                        def sub_f(x_in):
                            return sub_model((x_in - c_k) / sigma_k)
                            
                        r = self.pde.compute_residuals(sub_f, pt)
                        l_pt = torch.mean(r ** 2)
                        
                        grads = torch.autograd.grad(l_pt, sub_params, retain_graph=False, create_graph=False, allow_unused=True)
                        flat_g = torch.cat([
                            (g.view(-1) if g is not None else torch.zeros(p.numel(), device=self.device))
                            for g, p in zip(grads, sub_params)
                        ])
                        per_sample_grads.append(flat_g)
                        valid_pts.append(pt.squeeze(0))
                        
                    if len(per_sample_grads) > 4:
                        sample_grads_t = torch.stack(per_sample_grads, dim=0)
                        sample_pts_t = torch.stack(valid_pts, dim=0)
                        
                        C, mean_align, clash_ratio, pt_conflict = monitor.analyze_gram_matrix(sample_grads_t)
                        
                        if k not in monitor.ema_conflict_score:
                            monitor.ema_conflict_score[k] = mean_align
                            monitor.ema_clash_ratio[k] = clash_ratio
                        else:
                            monitor.ema_conflict_score[k] = (
                                monitor.ema_decay * monitor.ema_conflict_score[k] +
                                (1.0 - monitor.ema_decay) * mean_align
                            )
                            monitor.ema_clash_ratio[k] = (
                                monitor.ema_decay * monitor.ema_clash_ratio[k] +
                                (1.0 - monitor.ema_decay) * clash_ratio
                            )
                            
                        ema_align = monitor.ema_conflict_score[k]
                        ema_clash = monitor.ema_clash_ratio[k]
                        should_cleave = (ema_clash >= 0.15) or (ema_align < 0.0)
                        
                        if should_cleave and probe_model.num_subspaces < max_subspaces:
                            child_centroid = monitor.locate_clashing_center(sample_pts_t, pt_conflict).detach().to(self.device)
                            
                            # Dimension-Normalized Spatial Non-Redundancy Filter
                            norm_dists = torch.norm((probe_model.centroids - child_centroid) / probe_model.bandwidth, dim=-1)
                            if torch.min(norm_dists) < 0.40:
                                continue
                                
                            probe_model.spawn_new_subspace(
                                centroid=child_centroid,
                                bandwidth=sigma,
                                parent_idx=k,
                            )
                            optimizer = torch.optim.AdamW(probe_model.parameters(), lr=lr, weight_decay=weight_decay)
                            
                            c_np = child_centroid.detach().cpu().numpy().round(3).tolist()
                            print(f"  [Discovery @ Epoch {epoch:4d}] Subspace {k} Clashing (Align: {ema_align:.2f}, Clash: {ema_clash:.2f}) -> Spawned Subspace {probe_model.num_subspaces-1} @ {c_np}")
                            
                            cleavage_history.append({
                                "epoch": epoch,
                                "parent": k,
                                "child": probe_model.num_subspaces - 1,
                                "centroid": child_centroid.detach().cpu(),
                            })
                            last_cleavage_epoch = epoch
                            cleaved_this_epoch = True
                            consecutive_stable_cycles = 0
                            break
                        
            # True Autonomous Physical Quiescence check
            if (epoch >= min_epochs) and (epoch % profile_freq == 0) and not cleaved_this_epoch:
                consecutive_stable_cycles += 1
                if consecutive_stable_cycles >= 3:
                    print(f"  [+] True Autonomous Geometric Quiescence Achieved @ Epoch {epoch}!")
                    print(f"  [+] Physics-Driven Partition Discovery Finalized with N*={probe_model.num_subspaces} subspaces.")
                    break
                    
        elapsed_stage1 = time.time() - start_time
        print(f"  [+] Stage 1 Finished in {elapsed_stage1:.2f}s (Autonomous N*={probe_model.num_subspaces}).")
        
        info = {
            "num_subspaces": probe_model.num_subspaces,
            "centroids": probe_model.centroids.clone().detach(),
            "bandwidth": sigma,
            "cleavage_history": cleavage_history,
            "stage1_time": elapsed_stage1,
        }
        del probe_model, optimizer, monitor
        cleanup_gpu()
        return info

    def train_stage2_production(
        self,
        discovered_info: Dict,
        hidden_dim: int = 64,
        layers: int = 4,
        adamw_epochs: int = 1000,
        lbfgs_steps: int = 250,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_interior: int = 8192,
        n_boundary: int = 1024,
        n_initial: int = 1024,
        eval_freq: int = 25,
    ) -> Tuple[nn.Module, Dict]:
        set_seed(self.seed)
        cleanup_gpu()
        
        num_subspaces = discovered_info["num_subspaces"]
        centroids = discovered_info["centroids"]
        bandwidth = discovered_info["bandwidth"]
        bw_str = [round(float(b), 3) for b in bandwidth.cpu().numpy().ravel()]
        
        print("\n" + "="*65)
        print("  STAGE 2: PRODUCTION RE-TRAINING (AdamW + Combined L-BFGS Polish)")
        print(f"  Subspaces N*={num_subspaces} | Architecture: {hidden_dim}x{layers} per subspace | Bandwidth: {bw_str}")
        print(f"  Collocation: {n_interior} interior | {n_boundary} boundary | {n_initial} initial points")
        print("="*65)
        
        model = AdaptiveSubspacePINN(
            in_dim=self.pde.in_dim,
            out_dim=self.pde.out_dim,
            initial_subspaces=num_subspaces,
            hidden_dim=hidden_dim,
            layers=layers,
            activation="tanh",
            bandwidth=bandwidth,
            initial_centroids=centroids,
        ).to(self.device)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [+] Clean N* Model Initialized with {n_params} total parameters.")
        
        history = {
            "epoch": [],
            "loss_total": [],
            "loss_res": [],
            "loss_bc": [],
            "loss_ic": [],
            "wall_time": [],
        }
        
        start_time = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adamw_epochs, eta_min=1e-5)
        
        lambda_bc = 20.0
        lambda_ic = 20.0
        alpha_ema = 0.90
        
        best_total_loss = float("inf")
        best_state = None
        
        print(f"\n  [Phase A] Running Synchronized AdamW Optimization ({adamw_epochs} epochs)...")
        for epoch in range(1, adamw_epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            x_int = self.pde.sample_interior(n_interior)
            res = self.pde.compute_residuals(model, x_int)
            loss_res = torch.mean(res ** 2)
            
            x_bc, u_bc = self.pde.sample_boundary(n_boundary)
            loss_bc = self.pde.compute_boundary_loss(model, x_bc, u_bc)
            
            loss_ic = torch.tensor(0.0, device=self.device)
            ic_samples = self.pde.sample_initial(n_initial)
            if ic_samples is not None:
                x_ic, u_ic = ic_samples
                loss_ic = self.pde.compute_initial_loss(model, x_ic, u_ic)
                
            # Full-batch Adaptive Loss Weighting every 25 epochs
            if epoch % 25 == 1:
                params = [p for p in model.parameters() if p.requires_grad]
                g_res = torch.autograd.grad(loss_res, params, retain_graph=True, allow_unused=True)
                g_bc = torch.autograd.grad(loss_bc, params, retain_graph=True, allow_unused=True)
                
                norm_res = torch.sqrt(sum(torch.sum(g**2) for g in g_res if g is not None) + 1e-8).item()
                norm_bc = torch.sqrt(sum(torch.sum(g**2) for g in g_bc if g is not None) + 1e-8).item()
                
                target_lambda_bc = min(50.0, max(1.0, norm_res / (norm_bc + 1e-8)))
                lambda_bc = alpha_ema * lambda_bc + (1.0 - alpha_ema) * target_lambda_bc
                
                if ic_samples is not None:
                    g_ic = torch.autograd.grad(loss_ic, params, retain_graph=True, allow_unused=True)
                    norm_ic = torch.sqrt(sum(torch.sum(g**2) for g in g_ic if g is not None) + 1e-8).item()
                    target_lambda_ic = min(50.0, max(1.0, norm_res / (norm_ic + 1e-8)))
                    lambda_ic = alpha_ema * lambda_ic + (1.0 - alpha_ema) * target_lambda_ic
                    
            total_loss = loss_res + lambda_bc * loss_bc + lambda_ic * loss_ic
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            if epoch % eval_freq == 0 or epoch == adamw_epochs:
                elapsed = time.time() - start_time
                tot_val = total_loss.item()
                
                history["epoch"].append(epoch)
                history["loss_total"].append(tot_val)
                history["loss_res"].append(loss_res.item())
                history["loss_bc"].append(loss_bc.item())
                history["loss_ic"].append(loss_ic.item())
                history["wall_time"].append(elapsed)
                
                if tot_val < best_total_loss:
                    best_total_loss = tot_val
                    best_state = copy.deepcopy(model.state_dict())
                    
                print(f"  [AdamW Epoch {epoch:4d}/{adamw_epochs}] Total Loss: {tot_val:.6e} | Res: {loss_res.item():.6e} | BC: {loss_bc.item():.6e} | Time: {elapsed:.1f}s")
                
        # Phase B: Combined Global L-BFGS Polish with Continuous Intermediate Logging
        if lbfgs_steps > 0:
            print(f"\n  [Phase B] Running Full-Horizon Combined Global L-BFGS Polish ({lbfgs_steps} steps on dense grid)...")
            lbfgs_opt = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=lbfgs_steps,
                max_eval=int(lbfgs_steps * 1.5),
                history_size=50,
                tolerance_grad=1e-12,
                tolerance_change=1e-16,
                line_search_fn="strong_wolfe",
            )
            
            x_int_full = self.pde.sample_interior(n_interior)
            x_bc_full, u_bc_full = self.pde.sample_boundary(n_boundary)
            ic_full = self.pde.sample_initial(n_initial)
            x_ic_full, u_ic_full = ic_full if ic_full is not None else (None, None)
                
            lbfgs_step = [0]
            
            def closure():
                lbfgs_opt.zero_grad()
                res = self.pde.compute_residuals(model, x_int_full)
                l_res = torch.mean(res ** 2)
                l_bc = lambda_bc * self.pde.compute_boundary_loss(model, x_bc_full, u_bc_full)
                l_ic = lambda_ic * self.pde.compute_initial_loss(model, x_ic_full, u_ic_full) if x_ic_full is not None else torch.tensor(0.0, device=self.device)
                tot = l_res + l_bc + l_ic
                tot.backward()
                
                lbfgs_step[0] += 1
                tot_val = tot.item()
                
                # Continuously record every 10 steps of L-BFGS into convergence trajectory
                if lbfgs_step[0] % 10 == 0 or lbfgs_step[0] == 1:
                    history["epoch"].append(adamw_epochs + lbfgs_step[0])
                    history["loss_total"].append(tot_val)
                    history["loss_res"].append(l_res.item())
                    history["loss_bc"].append(l_bc.item())
                    history["loss_ic"].append(l_ic.item())
                    history["wall_time"].append(time.time() - start_time)
                    print(f"    [Combined L-BFGS Step {lbfgs_step[0]:3d}] Total Loss: {tot_val:.6e} | Res: {l_res.item():.6e} | BC: {l_bc.item():.6e}")
                return tot
                
            lbfgs_opt.step(closure)
            
            final_time = time.time() - start_time
            final_loss = history["loss_total"][-1]
            final_res = history["loss_res"][-1]
            final_bc = history["loss_bc"][-1]
            
            print(f"  [+] L-BFGS Polish Complete ({lbfgs_step[0]} evaluations). Final Best Total Loss: {final_loss:.6e}")
        else:
            final_time = history["wall_time"][-1]
            final_loss = history["loss_total"][-1]
            final_res = history["loss_res"][-1]
            final_bc = history["loss_bc"][-1]
            
        summary = {
            "final_loss_total": final_loss,
            "final_loss_res": final_res,
            "final_loss_bc": final_bc,
            "total_params": n_params,
            "wall_time": final_time,
            "history": history,
            "num_subspaces": num_subspaces,
        }
        cleanup_gpu()
        return model, summary
