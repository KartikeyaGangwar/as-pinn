import copy
import time
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics.base_pde import BasePDE


class MonolithicMLP(nn.Module):
    """Standard Multilayer Perceptron (MLP) for PINNs."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 96,
        layers: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        act_cls = nn.Tanh if activation.lower() == "tanh" else nn.GELU
        net = [nn.Linear(in_dim, hidden_dim), act_cls()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        net.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticFBPINN(nn.Module):
    """Static Finite Basis PINN (FBPINN) with fixed cartesian grid domain decomposition."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bounds: Tuple[Tuple[float, float], ...],
        grid_shape: Tuple[int, ...],
        hidden_dim: int = 44,
        layers: int = 3,
        activation: str = "tanh",
        overlap: float = 0.25,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_shape = grid_shape
        
        # Grid centroids and radii
        coords = [np.linspace(b[0], b[1], g) for b, g in zip(bounds, grid_shape)]
        mesh = np.meshgrid(*coords, indexing="ij")
        centroids = np.stack([m.ravel() for m in mesh], axis=1) # [K, in_dim]
        
        self.register_buffer("centroids", torch.tensor(centroids, dtype=torch.float32))
        self.num_subdomains = centroids.shape[0]
        
        dx = np.array([(b[1] - b[0]) / max(1, g - 1) if g > 1 else (b[1] - b[0]) for b, g in zip(bounds, grid_shape)])
        self.sigma = float(np.linalg.norm(dx) * (1.0 + overlap) * 0.5)
        
        act_cls = nn.Tanh if activation.lower() == "tanh" else nn.GELU
        self.subdomains = nn.ModuleList()
        for _ in range(self.num_subdomains):
            net = [nn.Linear(in_dim, hidden_dim), act_cls()]
            for _ in range(layers - 2):
                net.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
            net.append(nn.Linear(hidden_dim, out_dim))
            self.subdomains.append(nn.Sequential(*net))

    def partition_of_unity(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        logits = -dist_sq / (2.0 * (self.sigma ** 2) + 1e-8)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        psi = self.partition_of_unity(x)
        u_global = torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
        for k in range(self.num_subdomains):
            x_local = (x - self.centroids[k]) / self.sigma
            u_global = u_global + psi[:, k:k+1] * self.subdomains[k](x_local)
        return u_global


def train_model_adamw_lbfgs(
    pde,
    model: nn.Module,
    adamw_epochs: int = 1000,
    lbfgs_steps: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    n_interior: int = 8192,
    n_boundary: int = 1024,
    n_initial: int = 1024,
    eval_freq: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """Trains a model with AdamW followed by Second-Order L-BFGS Polish on dense collocation."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adamw_epochs, eta_min=1e-5)
    
    history = {"epoch": [], "loss": [], "rel_l2_err": [], "linf_err": [], "wall_time": []}
    start_time = time.time()
    
    lambda_bc = 20.0
    lambda_ic = 20.0
    alpha_ema = 0.90
    
    best_rel_l2 = float("inf")
    best_state = None
    
    for epoch in range(1, adamw_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        x_int = pde.sample_interior(n_interior)
        res = pde.compute_residuals(model, x_int)
        loss_res = torch.mean(res ** 2)
        
        x_bc, u_bc = pde.sample_boundary(n_boundary)
        loss_bc = pde.compute_boundary_loss(model, x_bc, u_bc)
        
        loss_ic = torch.tensor(0.0, device=device)
        ic_samples = pde.sample_initial(n_initial)
        if ic_samples is not None:
            x_ic, u_ic = ic_samples
            loss_ic = pde.compute_initial_loss(model, x_ic, u_ic)
            
        # Independent Gradient Norm Profiling every 25 epochs
        if epoch % 25 == 1:
            with torch.enable_grad():
                x_sub = pde.sample_interior(256)
                x_bc_sub, u_bc_sub = pde.sample_boundary(128)
                
                r_sub = pde.compute_residuals(model, x_sub)
                l_res_sub = torch.mean(r_sub ** 2)
                l_bc_sub = pde.compute_boundary_loss(model, x_bc_sub, u_bc_sub)
                
                params = [p for p in model.parameters() if p.requires_grad]
                g_res = torch.autograd.grad(l_res_sub, params, retain_graph=False, allow_unused=True)
                g_bc = torch.autograd.grad(l_bc_sub, params, retain_graph=False, allow_unused=True)
                
                norm_res = torch.sqrt(sum(torch.sum(g**2) for g in g_res if g is not None) + 1e-8).item()
                norm_bc = torch.sqrt(sum(torch.sum(g**2) for g in g_bc if g is not None) + 1e-8).item()
                target_lambda_bc = min(50.0, max(1.0, norm_res / (norm_bc + 1e-8)))
                lambda_bc = alpha_ema * lambda_bc + (1.0 - alpha_ema) * target_lambda_bc
                
                if ic_samples is not None:
                    ic_sub = pde.sample_initial(128)
                    if ic_sub is not None:
                        x_ic_sub, u_ic_sub = ic_sub
                        l_ic_sub = pde.compute_initial_loss(model, x_ic_sub, u_ic_sub)
                        g_ic = torch.autograd.grad(l_ic_sub, params, retain_graph=False, allow_unused=True)
                        norm_ic = torch.sqrt(sum(torch.sum(g**2) for g in g_ic if g is not None) + 1e-8).item()
                        target_lambda_ic = min(50.0, max(1.0, norm_res / (norm_ic + 1e-8)))
                        lambda_ic = alpha_ema * lambda_ic + (1.0 - alpha_ema) * target_lambda_ic
                        
        tot = loss_res + lambda_bc * loss_bc + lambda_ic * loss_ic
        tot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % eval_freq == 0 or epoch == adamw_epochs:
            model.eval()
            with torch.enable_grad():
                x_eval_int = pde.sample_interior(2048)
                r_eval = pde.compute_residuals(model, x_eval_int)
                l_res_eval = torch.mean(r_eval ** 2).item()
                
                x_eval_bc, u_eval_bc = pde.sample_boundary(512)
                l_bc_eval = pde.compute_boundary_loss(model, x_eval_bc, u_eval_bc).item()
                
                l_ic_eval = 0.0
                ic_eval = pde.sample_initial(512)
                if ic_eval is not None:
                    x_eval_ic, u_eval_ic = ic_eval
                    l_ic_eval = pde.compute_initial_loss(model, x_eval_ic, u_eval_ic).item()
                
                standard_composite_loss = l_res_eval + 20.0 * l_bc_eval + 20.0 * l_ic_eval

            rel_l2 = pde.compute_relative_l2_error(model, n_test=2000)
            linf = pde.compute_linf_error(model, n_test=2000)
            elapsed = time.time() - start_time
            
            history["epoch"].append(epoch)
            history["loss"].append(standard_composite_loss)
            history["rel_l2_err"].append(rel_l2)
            history["linf_err"].append(linf)
            history["wall_time"].append(elapsed)
            
            if rel_l2 < best_rel_l2:
                best_rel_l2 = rel_l2
                best_state = copy.deepcopy(model.state_dict())
                
    if lbfgs_steps > 0:
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
        x_int_f = pde.sample_interior(n_interior)
        x_bc_f, u_bc_f = pde.sample_boundary(n_boundary)
        ic_f = pde.sample_initial(n_initial)
        x_ic_f, u_ic_f = ic_f if ic_f is not None else (None, None)
        
        lbfgs_step = [0]
        
        def closure():
            lbfgs_opt.zero_grad()
            res = pde.compute_residuals(model, x_int_f)
            l_res = torch.mean(res ** 2)
            l_bc_eval = pde.compute_boundary_loss(model, x_bc_f, u_bc_f)
            l_ic_eval = pde.compute_initial_loss(model, x_ic_f, u_ic_f) if x_ic_f is not None else torch.tensor(0.0, device=device)
            
            tot_loss = l_res + lambda_bc * l_bc_eval + lambda_ic * l_ic_eval
            tot_loss.backward()
            
            lbfgs_step[0] += 1
            if lbfgs_step[0] % 10 == 0 or lbfgs_step[0] == 1:
                standard_lbfgs_loss = l_res.item() + 20.0 * l_bc_eval.item() + 20.0 * l_ic_eval.item()
                history["epoch"].append(adamw_epochs + lbfgs_step[0])
                history["loss"].append(standard_lbfgs_loss)
                history["wall_time"].append(time.time() - start_time)
            return tot_loss
            
        lbfgs_opt.step(closure)
        elapsed = time.time() - start_time
        
    return history


def train_pcgrad_baseline(
    pde,
    model: nn.Module,
    epochs: int = 1000,
    lr: float = 1e-3,
    eval_freq: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """PCGrad (Projecting Conflicting Gradients) Baseline with Standardized Composite Total Loss."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {"epoch": [], "loss": [], "rel_l2_err": [], "linf_err": [], "wall_time": []}
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        x_int = pde.sample_interior(4096)
        x_bc, u_bc = pde.sample_boundary(1024)
        ic_samples = pde.sample_initial(1024)
        
        res = pde.compute_residuals(model, x_int)
        l_res = torch.mean(res ** 2)
        l_bc = 20.0 * pde.compute_boundary_loss(model, x_bc, u_bc)
        
        losses = [l_res, l_bc]
        if ic_samples is not None:
            x_ic, u_ic = ic_samples
            l_ic = 20.0 * pde.compute_initial_loss(model, x_ic, u_ic)
            losses.append(l_ic)
            
        task_grads = []
        for l in losses:
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
            task_grads.append(grads)
            
        projected_grads = [copy.deepcopy(g) for g in task_grads]
        num_tasks = len(losses)
        
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    dot = sum(torch.sum(task_grads[i][k] * task_grads[j][k]) for k in range(len(task_grads[i])))
                    if dot < 0:
                        norm_sq = sum(torch.sum(task_grads[j][k] ** 2) for k in range(len(task_grads[j]))) + 1e-8
                        proj = dot / norm_sq
                        for k in range(len(projected_grads[i])):
                            projected_grads[i][k] -= proj * task_grads[j][k]
                            
        optimizer.zero_grad()
        for k, p in enumerate(model.parameters()):
            p.grad = sum(projected_grads[i][k] for i in range(num_tasks))
            
        optimizer.step()
        scheduler.step()
        
        if epoch % eval_freq == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            history["epoch"].append(epoch)
            history["loss"].append(sum(l.item() for l in losses))
            history["wall_time"].append(elapsed)
            
    return history


def train_cagrad_baseline(
    pde,
    model: nn.Module,
    epochs: int = 1000,
    lr: float = 1e-3,
    c: float = 0.5,
    eval_freq: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """CAGrad (Conflict-Averse Gradient Descent) Baseline with Standardized Composite Total Loss."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {"epoch": [], "loss": [], "rel_l2_err": [], "linf_err": [], "wall_time": []}
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        x_int = pde.sample_interior(4096)
        x_bc, u_bc = pde.sample_boundary(1024)
        ic_samples = pde.sample_initial(1024)
        
        res = pde.compute_residuals(model, x_int)
        l_res = torch.mean(res ** 2)
        l_bc = 20.0 * pde.compute_boundary_loss(model, x_bc, u_bc)
        
        losses = [l_res, l_bc]
        if ic_samples is not None:
            x_ic, u_ic = ic_samples
            l_ic = 20.0 * pde.compute_initial_loss(model, x_ic, u_ic)
            losses.append(l_ic)
            
        task_grads = []
        for l in losses:
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            flat_g = torch.cat([(p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)) for p in model.parameters()])
            task_grads.append(flat_g)
            
        G = torch.stack(task_grads, dim=0)
        g_mean = torch.mean(G, dim=0)
        GG = torch.mm(G, G.t())
        g_0 = torch.mean(GG, dim=1)
        
        alpha = torch.softmax(-g_0 / (torch.norm(g_mean) * c + 1e-8), dim=0)
        g_cagrad = torch.mv(G.t(), alpha)
        
        optimizer.zero_grad()
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.grad = g_cagrad[offset:offset+numel].view_as(p)
            offset += numel
            
        optimizer.step()
        scheduler.step()
        
        if epoch % eval_freq == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            history["epoch"].append(epoch)
            history["loss"].append(sum(l.item() for l in losses))
            history["wall_time"].append(elapsed)
            
    return history
