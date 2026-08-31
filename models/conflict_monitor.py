import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

class ContinuousConflictMonitor:
    """
    Continuous Intra-Subspace Gradient Conflict Monitor for AS-PINN.
    
    Uses PyTorch's native `torch.func` (vmap + grad / functional_call) to compute
    batched per-sample parameter gradients efficiently on GPU, constructing the
    intra-subspace pairwise cosine similarity Gram matrix C in O(1) vectorized kernel passes.
    
    When destructive interference is sustained (EMA alignment < threshold), it pinpoints
    the spatial coordinate cluster exhibiting the highest conflict to spawn a new subspace.
    """
    def __init__(
        self,
        threshold: float = 0.10,
        ema_decay: float = 0.85,
        min_clash_ratio: float = 0.25,
        in_dim: int = 2,
    ):
        """
        Args:
            threshold: Base cosine similarity threshold below which conflict triggers cleavage.
            ema_decay: Exponential moving average factor.
            min_clash_ratio: Minimum fraction of pairwise points with negative alignment (cos < 0).
            in_dim: Spatial dimension d for dimension-normalized conflict detection.
        """
        # Dimension-dependent scaling: in R^d, random vectors have expected cosine O(1/sqrt(d))
        dim_factor = (in_dim / 2.0) ** 0.5 if in_dim >= 2 else 1.0
        self.threshold = threshold / dim_factor
        self.min_clash_ratio = max(0.15, min_clash_ratio / dim_factor)
        self.ema_decay = ema_decay
        self.in_dim = in_dim
        self.ema_conflict_score: Dict[int, float] = {}
        self.ema_clash_ratio: Dict[int, float] = {}
        self.history: Dict[int, List[Dict[str, float]]] = {}

    def compute_per_point_gradients_vmap(
        self,
        subspace: nn.Module,
        x_batch: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Vectorized per-sample gradient computation using torch.func.
        
        Args:
            subspace: SubspaceMLP module (Theta_k).
            x_batch: [M, in_dim] batch of collocation points.
            loss_fn: Callable (u_pred, x_point) -> scalar loss for that point.
            
        Returns:
            G: [M, P] tensor containing per-point flattened parameter gradients.
        """
        params = dict(subspace.named_parameters())
        param_names = list(params.keys())
        
        # Single-point forward and loss calculation
        def single_point_eval(param_dict, x_single):
            # x_single shape: [in_dim] -> forward expects [1, in_dim]
            u_single = torch.func.functional_call(
                subspace, param_dict, (x_single.unsqueeze(0),)
            ).squeeze(0)
            return loss_fn(u_single, x_single)

        # Compute gradient w.r.t params for 1 sample
        grad_single_fn = torch.func.grad(single_point_eval, argnums=0)
        
        # Vectorize over the batch dimension of x_batch (in_dims: (None, 0))
        vmap_grad_fn = torch.func.vmap(grad_single_fn, in_dims=(None, 0))
        
        # grads_dict: dict of {name: Tensor[M, *param_shape]}
        grads_dict = vmap_grad_fn(params, x_batch)
        
        # Flatten and concatenate per-point parameter gradients into [M, P]
        flat_list = [grads_dict[name].reshape(x_batch.shape[0], -1) for name in param_names]
        G = torch.cat(flat_list, dim=1) # [M, P]
        return G

    def compute_per_point_pde_gradients_vmap(
        self,
        subspace: nn.Module,
        x_batch: torch.Tensor,
        pde_residual_fn: Callable[[Callable[[torch.Tensor], torch.Tensor], torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Vectorized per-sample gradient computation for full differential PDE operators.
        
        Args:
            subspace: SubspaceMLP module.
            x_batch: [M, in_dim] collocation points.
            pde_residual_fn: Callable(u_fn, x_single) -> scalar squared residual (r(x))^2.
        """
        params = dict(subspace.named_parameters())
        param_names = list(params.keys())

        def single_point_eval(param_dict, x_single):
            # Define local model mapping x -> u
            def u_fn(x):
                # x shape: [in_dim]
                return torch.func.functional_call(
                    subspace, param_dict, (x.unsqueeze(0),)
                ).squeeze(0)
            
            return pde_residual_fn(u_fn, x_single)

        grad_single_fn = torch.func.grad(single_point_eval, argnums=0)
        vmap_grad_fn = torch.func.vmap(grad_single_fn, in_dims=(None, 0))
        grads_dict = vmap_grad_fn(params, x_batch)
        
        flat_list = [grads_dict[name].reshape(x_batch.shape[0], -1) for name in param_names]
        G = torch.cat(flat_list, dim=1)
        return G

    def compute_per_point_gradients_autograd_fallback(
        self,
        subspace: nn.Module,
        x_batch: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Fallback implementation using standard autograd loop (for validation and debugging).
        """
        subspace_params = [p for p in subspace.parameters() if p.requires_grad]
        M = x_batch.shape[0]
        per_point_grads = []
        
        for i in range(M):
            xi = x_batch[i:i+1].clone().detach().requires_grad_(True)
            ui = subspace(xi).squeeze(0)
            loss = loss_fn(ui, xi.squeeze(0))
            grads = torch.autograd.grad(loss, subspace_params, retain_graph=False, create_graph=False)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            per_point_grads.append(flat_grad)
            
        return torch.stack(per_point_grads)

    def analyze_gram_matrix(
        self,
        G: torch.Tensor,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        """
        Analyzes the gradient Gram matrix and cosine similarity structure.
        
        Args:
            G: [M, P] per-point gradient matrix.
            eps: Numerical stability constant.
            
        Returns:
            C: [M, M] pairwise cosine similarity matrix.
            mean_alignment: Mean off-diagonal cosine similarity.
            clash_ratio: Fraction of pairwise cosine similarities < 0.
            point_conflict_scores: [M] sum of negative alignments per point.
        """
        M, P = G.shape
        if M < 2:
            C = torch.ones((M, M), device=G.device)
            return C, 1.0, 0.0, torch.zeros(M, device=G.device)

        # Normalize gradients: G_norm = G / ||G||_2
        gnorms = torch.norm(G, p=2, dim=1, keepdim=True).clamp_min(eps)
        G_norm = G / gnorms
        
        # Pairwise Cosine Similarity Matrix: C = G_norm @ G_norm^T
        C = torch.mm(G_norm, G_norm.t()).clamp(-1.0, 1.0)
        
        # Off-diagonal mask
        mask = ~torch.eye(M, dtype=torch.bool, device=G.device)
        off_diags = C[mask]
        
        mean_alignment = off_diags.mean().item()
        clash_ratio = (off_diags < 0.0).float().mean().item()
        
        # Per-point conflict index: how much does point i clash with other points?
        # Sum of negative alignments or mean alignment with others
        C_no_diag = C.masked_fill(~mask, 0.0)
        # Negative part of cosine similarities
        neg_C = torch.clamp(C_no_diag, max=0.0)
        point_conflict_scores = -neg_C.sum(dim=1) # High positive value = high conflict
        
        return C, mean_alignment, clash_ratio, point_conflict_scores

    def locate_clashing_center(
        self,
        x_batch: torch.Tensor,
        point_conflict_scores: torch.Tensor,
        top_k_fraction: float = 0.35,
    ) -> torch.Tensor:
        """
        Identifies the spatial cluster centroid where destructive gradient interference is concentrated.
        
        Args:
            x_batch: [M, in_dim] collocation points.
            point_conflict_scores: [M] conflict severity per point.
            top_k_fraction: Fraction of top clashing points to average over.
            
        Returns:
            centroid: [in_dim] spatial coordinate for newly spawned subspace.
        """
        M = x_batch.shape[0]
        k = max(1, int(M * top_k_fraction))
        
        # If all scores are 0 (no conflict), return the point with highest gradient variance or mean
        if point_conflict_scores.max() == 0:
            return x_batch.mean(dim=0)
            
        top_indices = torch.topk(point_conflict_scores, k=k).indices
        clashing_points = x_batch[top_indices]
        clashing_weights = point_conflict_scores[top_indices].clamp_min(1e-6)
        
        # Weighted centroid
        weighted_center = (clashing_points * clashing_weights.unsqueeze(-1)).sum(dim=0) / clashing_weights.sum()
        return weighted_center

    def profile_subspace(
        self,
        model,
        subspace_idx: int,
        x_batch: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        use_vmap: bool = True,
    ) -> Tuple[bool, Dict[str, float], Optional[torch.Tensor]]:
        """
        Profiles a specific subspace for intra-task gradient clash.
        
        Returns:
            (should_cleave, metrics_dict, suggested_centroid)
        """
        subspace = model.subspaces[subspace_idx]
        
        # 1. Compute per-point gradients
        if use_vmap:
            try:
                G = self.compute_per_point_gradients_vmap(subspace, x_batch, loss_fn)
            except Exception as e:
                # In case of nested backward limitations in complex dynamic ops, fallback
                G = self.compute_per_point_gradients_autograd_fallback(subspace, x_batch, loss_fn)
        else:
            G = self.compute_per_point_gradients_autograd_fallback(subspace, x_batch, loss_fn)
            
        # 2. Analyze Gram matrix & Cosine similarity
        C, mean_align, clash_ratio, pt_conflict = self.analyze_gram_matrix(G)
        
        # 3. Update Exponential Moving Average (EMA)
        if subspace_idx not in self.ema_conflict_score:
            self.ema_conflict_score[subspace_idx] = mean_align
            self.ema_clash_ratio[subspace_idx] = clash_ratio
        else:
            self.ema_conflict_score[subspace_idx] = (
                self.ema_decay * self.ema_conflict_score[subspace_idx] +
                (1.0 - self.ema_decay) * mean_align
            )
            self.ema_clash_ratio[subspace_idx] = (
                self.ema_decay * self.ema_clash_ratio[subspace_idx] +
                (1.0 - self.ema_decay) * clash_ratio
            )
            
        cur_ema_align = self.ema_conflict_score[subspace_idx]
        cur_ema_clash = self.ema_clash_ratio[subspace_idx]
        
        metrics = {
            "instant_alignment": mean_align,
            "ema_alignment": cur_ema_align,
            "instant_clash_ratio": clash_ratio,
            "ema_clash_ratio": cur_ema_clash,
        }
        
        if subspace_idx not in self.history:
            self.history[subspace_idx] = []
        self.history[subspace_idx].append(metrics)
        
        # 4. Cleavage Condition:
        # Sustained severe negative alignment and high clash ratio
        should_cleave = (cur_ema_align < self.threshold) and (cur_ema_clash >= self.min_clash_ratio)
        
        suggested_center = None
        if should_cleave:
            suggested_center = self.locate_clashing_center(x_batch, pt_conflict)
            
        return should_cleave, metrics, suggested_center

    def profile_from_gradient_matrix(
        self,
        subspace_idx: int,
        G: torch.Tensor,
        x_batch: torch.Tensor,
    ) -> Tuple[bool, Dict[str, float], Optional[torch.Tensor]]:
        """
        Profiles a subspace directly from a precomputed [M, P] gradient matrix G.
        
        Returns:
            (should_cleave, metrics_dict, suggested_centroid)
        """
        C, mean_align, clash_ratio, pt_conflict = self.analyze_gram_matrix(G)
        
        if subspace_idx not in self.ema_conflict_score:
            self.ema_conflict_score[subspace_idx] = mean_align
            self.ema_clash_ratio[subspace_idx] = clash_ratio
        else:
            self.ema_conflict_score[subspace_idx] = (
                self.ema_decay * self.ema_conflict_score[subspace_idx] +
                (1.0 - self.ema_decay) * mean_align
            )
            self.ema_clash_ratio[subspace_idx] = (
                self.ema_decay * self.ema_clash_ratio[subspace_idx] +
                (1.0 - self.ema_decay) * clash_ratio
            )
            
        cur_ema_align = self.ema_conflict_score[subspace_idx]
        cur_ema_clash = self.ema_clash_ratio[subspace_idx]
        
        metrics = {
            "instant_alignment": mean_align,
            "ema_alignment": cur_ema_align,
            "instant_clash_ratio": clash_ratio,
            "ema_clash_ratio": cur_ema_clash,
        }
        
        if subspace_idx not in self.history:
            self.history[subspace_idx] = []
        self.history[subspace_idx].append(metrics)
        
        should_cleave = (cur_ema_align < self.threshold) and (cur_ema_clash >= self.min_clash_ratio)
        suggested_center = None
        if should_cleave:
            suggested_center = self.locate_clashing_center(x_batch, pt_conflict)
            
        return should_cleave, metrics, suggested_center

    def reset_subspace(self, subspace_idx: int):
        """Resets the conflict monitor for a cleaved subspace."""
        self.ema_conflict_score[subspace_idx] = 0.5
        self.ema_clash_ratio[subspace_idx] = 0.0

