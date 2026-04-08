"""
Multi-Objective Pareto GFlowNet (MOGFN) — Publication Improvement #7
====================================================================
Based on: Jain et al. "Multi-Objective GFlowNets" (ICML 2023)

Preference-conditioned (MOGFN-PC) architecture that:
  1. Samples preferences ω ~ Dir(α) from Dirichlet distribution
  2. Conditions the policy network on ω via feature concatenation
  3. Scalarizes multi-objective reward: R(x; ω) = Π_i R_i(x)^{ω_i}
  4. Trains the GFlowNet to generate diverse Pareto-optimal solutions
  
This enables *single-model* generation of the full Pareto front by
varying ω at inference time, rather than training separate models.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class MOGFNPreferenceEncoder(nn.Module):
    """
    Encodes preference vector ω ∈ Δ^{K-1} (K-simplex) into a
    fixed-size representation that is concatenated to policy input.
    
    Uses 2-layer MLP with ReLU, following Jain et al. ICML 2023.
    """
    
    def __init__(self, num_objectives: int = 3, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.num_objectives = num_objectives
        self.encoder = nn.Sequential(
            nn.Linear(num_objectives, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Encode preference vector.
        
        Args:
            omega: (batch_size, num_objectives) preference weights on simplex
        
        Returns:
            (batch_size, output_dim) encoded preferences
        """
        return self.encoder(omega)


class ParetoFront:
    """
    Maintains a Pareto front of non-dominated solutions.
    
    A solution x dominates y iff x_i >= y_i for all i AND x_j > y_j for some j.
    The Pareto front contains all non-dominated solutions.
    """
    
    def __init__(self, num_objectives: int = 3, max_size: int = 500):
        self.num_objectives = num_objectives
        self.max_size = max_size
        self.solutions = []  # List of (smiles, objectives_dict)
    
    def is_dominated(self, obj_values: List[float], existing: List[float]) -> bool:
        """Check if obj_values is dominated by existing."""
        return all(e >= o for e, o in zip(existing, obj_values)) and \
               any(e > o for e, o in zip(existing, obj_values))
    
    def add(self, smiles: str, objectives: Dict[str, float]) -> bool:
        """
        Add solution to Pareto front if non-dominated.
        
        Returns True if solution was added (is Pareto-optimal).
        """
        obj_values = [objectives.get('s_bio', 0), objectives.get('s_mech', 0), objectives.get('s_syn', 0)]
        
        # Check if dominated by any existing solution
        to_remove = []
        for i, (_, existing_obj) in enumerate(self.solutions):
            existing_values = [existing_obj.get('s_bio', 0), existing_obj.get('s_mech', 0), existing_obj.get('s_syn', 0)]
            if self.is_dominated(obj_values, existing_values):
                return False  # New solution is dominated
            if self.is_dominated(existing_values, obj_values):
                to_remove.append(i)
        
        # Remove dominated solutions
        for i in sorted(to_remove, reverse=True):
            self.solutions.pop(i)
        
        # Add new solution
        self.solutions.append((smiles, objectives))
        
        # If too large, remove lowest hypervolume contributor
        if len(self.solutions) > self.max_size:
            self.solutions.pop(0)  # Remove oldest
        
        return True
    
    def get_hypervolume(self, reference_point: Optional[List[float]] = None) -> float:
        """
        Compute hypervolume indicator of current Pareto front.
        
        Uses simple 2D/3D approximation (exact HV is NP-hard for >3 objectives).
        """
        if not self.solutions:
            return 0.0
        
        if reference_point is None:
            reference_point = [0.0] * self.num_objectives
        
        # For 3 objectives, use simple dominated hypervolume approximation
        hv = 0.0
        for _, obj in self.solutions:
            vals = [obj.get('s_bio', 0) - reference_point[0],
                    obj.get('s_mech', 0) - reference_point[1],
                    obj.get('s_syn', 0) - reference_point[2]]
            if all(v > 0 for v in vals):
                hv += vals[0] * vals[1] * vals[2]
        
        return hv
    
    @property
    def size(self) -> int:
        return len(self.solutions)
    
    def get_front_as_array(self) -> np.ndarray:
        """Get Pareto front as (N, 3) numpy array."""
        if not self.solutions:
            return np.zeros((0, 3))
        return np.array([
            [obj.get('s_bio', 0), obj.get('s_mech', 0), obj.get('s_syn', 0)]
            for _, obj in self.solutions
        ])
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of the Pareto front."""
        if not self.solutions:
            return {'size': 0, 'hypervolume': 0.0}
        
        arr = self.get_front_as_array()
        return {
            'size': len(self.solutions),
            'hypervolume': self.get_hypervolume(),
            'mean_bio': float(arr[:, 0].mean()),
            'mean_mech': float(arr[:, 1].mean()),
            'mean_syn': float(arr[:, 2].mean()),
            'max_bio': float(arr[:, 0].max()),
            'max_mech': float(arr[:, 1].max()),
            'max_syn': float(arr[:, 2].max()),
        }


def sample_dirichlet_preferences(
    batch_size: int = 1,
    num_objectives: int = 3,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Sample preference vectors from Dirichlet distribution.
    
    α=1.0: uniform over simplex (default — explores all trade-offs)
    α>1.0: concentrates toward center (balanced preferences)
    α<1.0: concentrates toward vertices (extreme preferences)
    
    Args:
        batch_size: number of preference vectors
        num_objectives: number of objectives (default: 3 = bio, mech, syn)
        alpha: Dirichlet concentration parameter
    
    Returns:
        (batch_size, num_objectives) preference vectors on simplex
    """
    alphas = np.full(num_objectives, alpha)
    samples = np.random.dirichlet(alphas, size=batch_size)
    return torch.FloatTensor(samples)


def scalarize_reward(
    objectives: Dict[str, float],
    omega: np.ndarray,
    method: str = 'weighted_power',
) -> float:
    """
    Scalarize multi-objective reward using preference vector.
    
    Methods:
        'weighted_power': R(x;ω) = Π_i R_i(x)^{ω_i}  (Jain et al.)
        'weighted_sum':   R(x;ω) = Σ_i ω_i * R_i(x)
        'tchebycheff':    R(x;ω) = max_i ω_i * |R_i(x) - z*_i|
    
    Args:
        objectives: dict with s_bio, s_mech, s_syn scores
        omega: preference weights (sum to 1)
        method: scalarization method
    
    Returns:
        Scalarized reward value
    """
    s_bio = max(objectives.get('s_bio', 0.0), 1e-8)
    s_mech = max(objectives.get('s_mech', 0.0), 1e-8)
    s_syn = max(objectives.get('s_syn', 0.0), 1e-8)
    
    if method == 'weighted_power':
        # Geometric scalarization: Π R_i^{ω_i}
        return float(s_bio ** omega[0] * s_mech ** omega[1] * s_syn ** omega[2])
    elif method == 'weighted_sum':
        return float(omega[0] * s_bio + omega[1] * s_mech + omega[2] * s_syn)
    elif method == 'tchebycheff':
        # Uses ideal point z* = (1, 1, 1) 
        return float(-max(omega[0] * abs(1.0 - s_bio),
                          omega[1] * abs(1.0 - s_mech),
                          omega[2] * abs(1.0 - s_syn)))
    else:
        raise ValueError(f"Unknown scalarization method: {method}")


def create_mogfn_config(
    num_objectives: int = 3,
    dirichlet_alpha: float = 1.0,
    pref_hidden_dim: int = 64,
    pref_output_dim: int = 32,
    pareto_max_size: int = 500,
    scalarization: str = 'weighted_power',
) -> Dict:
    """Create configuration for MOGFN extension."""
    return {
        'mogfn_enabled': True,
        'num_objectives': num_objectives,
        'dirichlet_alpha': dirichlet_alpha,
        'pref_hidden_dim': pref_hidden_dim,
        'pref_output_dim': pref_output_dim,
        'pareto_max_size': pareto_max_size,
        'scalarization_method': scalarization,
    }
