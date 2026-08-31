import abc
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class BasePDE(abc.ABC):
    """
    Abstract Base Class for SciML Benchmark PDE Systems.
    """
    def __init__(self, in_dim: int, out_dim: int, device: Optional[torch.device] = None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples collocation points in the interior domain."""
        pass

    @abc.abstractmethod
    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples boundary points and evaluates exact/prescribed boundary values."""
        pass

    def sample_initial(self, n_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Samples t=0 initial condition points for time-dependent PDEs."""
        return None

    @abc.abstractmethod
    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """Computes PDE interior residuals r(x)."""
        pass

    def compute_boundary_loss(
        self, model: nn.Module, x_bc: torch.Tensor, u_bc_exact: torch.Tensor
    ) -> torch.Tensor:
        """Computes boundary condition Mean Squared Error."""
        u_pred = model(x_bc)
        return torch.mean((u_pred - u_bc_exact) ** 2)

    def compute_initial_loss(
        self, model: nn.Module, x_ic: torch.Tensor, u_ic_exact: torch.Tensor
    ) -> torch.Tensor:
        """Computes initial condition Mean Squared Error."""
        u_pred = model(x_ic)
        return torch.mean((u_pred - u_ic_exact) ** 2)

    @abc.abstractmethod
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Returns exact ground-truth solution for validation/benchmarking."""
        pass

    def compute_relative_l2_error(
        self, model: nn.Module, x_test: Optional[torch.Tensor] = None, n_test: int = 2000
    ) -> float:
        """Computes relative L2 error: ||u_pred - u_exact||_2 / ||u_exact||_2."""
        if x_test is None:
            x_test = self.sample_interior(n_test)
        
        with torch.no_grad():
            u_pred = model(x_test)
            u_exact = self.exact_solution(x_test)
            l2_err = torch.norm(u_pred - u_exact, p=2)
            l2_ref = torch.norm(u_exact, p=2).clamp_min(1e-8)
            rel_error = (l2_err / l2_ref).item()
            
        return rel_error

    def compute_linf_error(
        self, model: nn.Module, x_test: Optional[torch.Tensor] = None, n_test: int = 2000
    ) -> float:
        """Computes maximum absolute error: max |u_pred - u_exact|."""
        if x_test is None:
            x_test = self.sample_interior(n_test)
            
        with torch.no_grad():
            u_pred = model(x_test)
            u_exact = self.exact_solution(x_test)
            max_err = torch.max(torch.abs(u_pred - u_exact)).item()
            
        return max_err
