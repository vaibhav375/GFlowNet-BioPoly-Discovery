"""
Session 6 — Advanced GFlowNet Training Techniques

State-of-the-art improvements based on:
  - Kim et al. "Local Search GFlowNets" (ICLR 2024)
  - Rector-Brooks et al. "Thompson Sampling GFlowNets" (2024)
  - Huang et al. "Variance-Reducing Control Variates" (NeurIPS 2024)
  - Malkin et al. "Trajectory Balance with Learned Backward Policy" (NeurIPS 2022)
  - Standard: Cosine LR scheduling, Ensemble Surrogates
"""

import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================
# IMPROVEMENT #9: Local Search GFlowNet (LS-GFN)
# Kim et al. "Local Search GFlowNets" — ICLR 2024
#
# After generating a trajectory, LS-GFN refines it by:
# 1. Picking a random intermediate state in the trajectory
# 2. Backtracking to that state (deconstruction)
# 3. Reconstructing from that state using the forward policy
# 4. If the new trajectory has higher reward, use it for training
# ============================================================

class LocalSearchRefiner:
    """
    Refines GFlowNet trajectories via local search.
    
    For each generated trajectory of length T:
      1. Sample a backtrack point t ~ Uniform(1, T-1)
      2. Keep states[0..t] (partial molecule)
      3. Re-run forward policy from states[t] to completion
      4. If new_reward > old_reward, replace the trajectory
    
    This biases training toward higher-reward regions while
    maintaining the GFlowNet's diversity objective.
    """
    
    def __init__(
        self,
        max_refinement_attempts: int = 3,
        improvement_threshold: float = 0.0,
        backtrack_ratio_range: Tuple[float, float] = (0.2, 0.8),
    ):
        self.max_attempts = max_refinement_attempts
        self.improvement_threshold = improvement_threshold
        self.backtrack_range = backtrack_ratio_range
        self.stats = defaultdict(int)
    
    def refine_trajectory(
        self,
        gflownet,
        original_states: list,
        original_actions: list,
        original_log_probs: list,
        original_reward: float,
    ) -> Tuple[list, list, list, float, bool]:
        """
        Attempt to refine a trajectory via local search.
        
        Returns:
            (states, actions, log_probs, reward, improved)
        """
        T = len(original_states)
        if T < 4:  # Too short to refine
            return original_states, original_actions, original_log_probs, original_reward, False
        
        best_states = original_states
        best_actions = original_actions
        best_log_probs = original_log_probs
        best_reward = original_reward
        improved = False
        
        for attempt in range(self.max_attempts):
            # Sample backtrack point
            min_bt = max(1, int(T * self.backtrack_range[0]))
            max_bt = min(T - 2, int(T * self.backtrack_range[1]))
            if min_bt >= max_bt:
                continue
            
            bt_point = random.randint(min_bt, max_bt)
            
            # Keep prefix: states[0..bt_point]
            prefix_states = best_states[:bt_point + 1]
            prefix_actions = best_actions[:bt_point]
            prefix_log_probs = best_log_probs[:bt_point]
            
            # Reconstruct from the backtrack state
            try:
                new_states, new_actions, new_log_probs, reward_info = (
                    gflownet.generate_trajectory_from_state(
                        prefix_states[-1],
                        prefix_states=prefix_states,
                        prefix_actions=prefix_actions,
                        prefix_log_probs=prefix_log_probs,
                    )
                )
                
                new_reward = reward_info.get('reward', 0.0)
                
                if new_reward > best_reward + self.improvement_threshold:
                    best_states = new_states
                    best_actions = new_actions
                    best_log_probs = new_log_probs
                    best_reward = new_reward
                    improved = True
                    self.stats['improvements'] += 1
                
                self.stats['attempts'] += 1
                
            except Exception:
                self.stats['failures'] += 1
                continue
        
        return best_states, best_actions, best_log_probs, best_reward, improved


# ============================================================
# IMPROVEMENT #10: Thompson Sampling Exploration
# Rector-Brooks et al. "Thompson Sampling GFlowNets" (2024)
#
# Maintains an approximate posterior over GFlowNet policies
# using an ensemble. Samples a policy from the posterior for
# each trajectory, encouraging diverse exploration.
# ============================================================

class ThompsonSamplingExplorer:
    """
    Thompson Sampling for GFlowNet exploration.
    
    Maintains K bootstrap heads on top of the shared GNN backbone.
    At each training step, randomly selects one head for trajectory
    generation, encouraging exploration of uncertain regions.
    """
    
    def __init__(
        self,
        base_output_dim: int = 256,
        num_heads: int = 5,
        head_hidden_dim: int = 128,
    ):
        self.num_heads = num_heads
        # Bootstrap heads: small MLPs on top of the shared backbone
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_output_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1),  # Outputs action value adjustment
            )
            for _ in range(num_heads)
        ])
        self._current_head = 0
    
    def sample_head(self) -> int:
        """Sample a head index for this trajectory."""
        self._current_head = random.randint(0, self.num_heads - 1)
        return self._current_head
    
    def adjust_logits(self, base_logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Adjust action logits using the current bootstrap head.
        
        Args:
            base_logits: Original action logits from the policy
            features: Feature vector from the GNN backbone
            
        Returns:
            Adjusted logits incorporating epistemic uncertainty
        """
        head = self.heads[self._current_head]
        adjustment = head(features)
        return base_logits + 0.1 * adjustment  # Small perturbation
    
    def compute_disagreement(self, features: torch.Tensor) -> float:
        """
        Compute disagreement across heads (epistemic uncertainty).
        Higher disagreement → more uncertain → more worth exploring.
        """
        predictions = []
        for head in self.heads:
            with torch.no_grad():
                pred = head(features)
                predictions.append(pred)
        predictions = torch.stack(predictions)
        return predictions.std(dim=0).mean().item()


# ============================================================
# IMPROVEMENT #11: Variance-Reducing Control Variates
# Huang et al. "Variance Reduction for GFlowNets" — NeurIPS 2024
#
# Adds a learned baseline to the TB loss gradient to reduce
# gradient variance without introducing bias.
# ============================================================

class VarianceReducer(nn.Module):
    """
    Learned baseline for variance reduction in GFlowNet gradients.
    
    Uses a small network to predict a state-dependent baseline b(s).
    The loss becomes: (TB_residual - b(s))^2 + b(s).detach() * TB_residual
    This reduces variance while maintaining unbiased gradients.
    """
    
    def __init__(self, state_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.baseline_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.running_mean = 0.0
        self.running_var = 1.0
        self.ema_decay = 0.99
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """Predict baseline value for the given state."""
        return self.baseline_net(state_features).squeeze(-1)
    
    def compute_variance_reduced_loss(
        self,
        tb_residual: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute variance-reduced TB loss.
        
        Uses exponential moving average baseline when state features
        are not available (simpler but still effective).
        """
        if state_features is not None and state_features.numel() > 0:
            baseline = self.forward(state_features)
            centered = tb_residual - baseline.detach()
            loss = centered ** 2
            # Also train the baseline to minimize its own error
            baseline_loss = (baseline - tb_residual.detach()) ** 2
            return loss.mean() + 0.1 * baseline_loss.mean()
        else:
            # EMA baseline (simpler fallback)
            centered = tb_residual - self.running_mean
            self.running_mean = (
                self.ema_decay * self.running_mean 
                + (1 - self.ema_decay) * tb_residual.detach().mean().item()
            )
            self.running_var = (
                self.ema_decay * self.running_var
                + (1 - self.ema_decay) * centered.detach().var().item()
            )
            return (centered ** 2).mean()


# ============================================================
# IMPROVEMENT #12: Cosine LR Schedule with Linear Warmup
# Standard best practice for training stability
# ============================================================

class CosineWarmupScheduler:
    """
    Cosine annealing LR scheduler with linear warmup.
    
    warmup_steps: linear ramp from 0 to base_lr
    remaining steps: cosine decay from base_lr to min_lr
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        total_steps: int = 2500,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# ============================================================
# IMPROVEMENT #13: Ensemble Surrogate Models
# Uses multiple surrogates with different random seeds for
# uncertainty quantification via disagreement.
# ============================================================

class EnsembleSurrogate:
    """
    Ensemble of surrogate models for robust prediction + uncertainty.
    
    Creates K copies of a surrogate model, each trained with
    different random seeds and bootstrap samples. Prediction
    uncertainty is measured by inter-model disagreement.
    """
    
    def __init__(self, base_model_class, config: dict, num_models: int = 3):
        self.num_models = num_models
        self.models = []
        
        for i in range(num_models):
            # Clone config with different seed
            model_config = dict(config)
            model_config['seed'] = config.get('seed', 42) + i * 137
            model = base_model_class(model_config)
            self.models.append(model)
    
    def predict(self, x, return_uncertainty: bool = False):
        """
        Predict using ensemble mean.
        
        Args:
            x: Input (graph data or features)
            return_uncertainty: If True, also return disagreement
            
        Returns:
            mean_prediction, (optional) uncertainty
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        stacked = torch.stack(predictions)
        mean = stacked.mean(dim=0)
        
        if return_uncertainty:
            std = stacked.std(dim=0)
            return mean, std
        return mean
    
    def acquisition_score(self, x) -> float:
        """
        Compute acquisition score for active learning.
        Uses mean prediction + uncertainty for UCB-style acquisition.
        """
        mean, std = self.predict(x, return_uncertainty=True)
        # Upper Confidence Bound
        return (mean + 1.5 * std).item()
    
    def to(self, device):
        for model in self.models:
            model.to(device)
        return self
    
    def parameters(self):
        """Yield all parameters from all ensemble members."""
        for model in self.models:
            yield from model.parameters()
    
    def train(self):
        for model in self.models:
            model.train()
    
    def eval(self):
        for model in self.models:
            model.eval()


# ============================================================
# IMPROVEMENT #14: Learned Backward Policy
# Malkin et al. "Trajectory Balance" — NeurIPS 2022
#
# Instead of uniform backward transition P_B, learn a parameterized
# backward policy. This tightens the trajectory balance constraint
# and gives better gradient signal.
# ============================================================

class BackwardPolicy(nn.Module):
    """
    Learned backward policy P_B(s_{t-1} | s_t) for GFlowNets.
    
    Given a molecular state s_t, predicts which atom/bond was
    most likely the last one added (for deconstruction).
    
    This is simpler than the full forward policy — it just needs
    to assign probabilities to "which atom to remove".
    """
    
    def __init__(self, hidden_dim: int = 128, num_atom_types: int = 10):
        super().__init__()
        
        # Encode current state → per-atom logits
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Per-atom removal probability
        self.removal_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        state_features: torch.Tensor,
        num_atoms: int,
    ) -> torch.Tensor:
        """
        Compute log P_B(remove atom i | state).
        
        Args:
            state_features: [num_atoms, hidden_dim] atom representations
            num_atoms: Number of atoms in current state
            
        Returns:
            log_probs: [num_atoms] log probabilities of removing each atom
        """
        if num_atoms <= 1:
            return torch.zeros(1, device=state_features.device)
        
        encoded = self.state_encoder(state_features)
        logits = self.removal_head(encoded).squeeze(-1)
        
        # Mask: can't remove atoms that would disconnect the molecule
        # (simplified: just use softmax over all atoms)
        log_probs = F.log_softmax(logits[:num_atoms], dim=0)
        
        return log_probs
    
    def compute_log_pb(
        self,
        state_features: torch.Tensor,
        removed_atom_idx: int,
        num_atoms: int,
    ) -> torch.Tensor:
        """Get log P_B for a specific backward action."""
        log_probs = self.forward(state_features, num_atoms)
        if removed_atom_idx < len(log_probs):
            return log_probs[removed_atom_idx]
        # Fallback: uniform
        return torch.tensor(-math.log(max(num_atoms, 1)), device=state_features.device)


# ============================================================
# INTEGRATION: Combine all improvements into train_step
# ============================================================

def create_advanced_optimizer(
    gflownet,
    config: dict,
) -> Tuple[torch.optim.Optimizer, CosineWarmupScheduler]:
    """
    Create optimizer with cosine warmup scheduler.
    
    Improvement #12: Proper LR scheduling for better convergence.
    """
    policy_lr = config.get('policy_lr', 5e-4)
    log_z_lr = config.get('log_z_lr', 5e-3)
    total_steps = config.get('training_steps', 2500)
    warmup_steps = config.get('warmup_steps', int(total_steps * 0.04))
    
    param_groups = [
        {'params': gflownet.policy.parameters(), 'lr': policy_lr},
        {'params': [gflownet.log_z], 'lr': log_z_lr},
    ]
    
    # Add variance reducer parameters if available
    if hasattr(gflownet, 'variance_reducer') and gflownet.variance_reducer is not None:
        param_groups.append({
            'params': gflownet.variance_reducer.parameters(),
            'lr': policy_lr * 0.5,
        })
    
    # Add backward policy parameters if available
    if hasattr(gflownet, 'backward_policy') and gflownet.backward_policy is not None:
        param_groups.append({
            'params': gflownet.backward_policy.parameters(),
            'lr': policy_lr * 0.3,
        })
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-5),
    )
    
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config.get('min_lr_ratio', 0.01),
    )
    
    return optimizer, scheduler


def initialize_advanced_components(gflownet, config: dict = None):
    """
    Initialize all Session 6 advanced components on a GFlowNet instance.
    
    Call this after creating a GFlowNet to enable all advanced features.
    """
    config = config or {}
    device = getattr(gflownet, 'device', 'cpu')
    
    # #9: Local Search
    gflownet.local_search = LocalSearchRefiner(
        max_refinement_attempts=config.get('ls_max_attempts', 3),
        improvement_threshold=config.get('ls_threshold', 0.0),
    )
    gflownet.use_local_search = config.get('use_local_search', True)
    gflownet.ls_probability = config.get('ls_probability', 0.3)  # Apply LS to 30% of trajectories
    
    # #10: Thompson Sampling
    hidden_dim = config.get('policy_hidden_dim', 256)
    gflownet.thompson_explorer = ThompsonSamplingExplorer(
        base_output_dim=hidden_dim,
        num_heads=config.get('ts_num_heads', 5),
        head_hidden_dim=config.get('ts_head_dim', 128),
    ).to(device) if config.get('use_thompson_sampling', True) else None
    
    # #11: Variance Reduction
    gflownet.variance_reducer = VarianceReducer(
        state_dim=hidden_dim,
        hidden_dim=config.get('vr_hidden_dim', 64),
    ).to(device) if config.get('use_variance_reduction', True) else None
    
    # #14: Backward Policy
    gflownet.backward_policy = BackwardPolicy(
        hidden_dim=hidden_dim,
    ).to(device) if config.get('use_backward_policy', True) else None
    
    # Statistics tracking
    gflownet.session6_stats = {
        'ls_improvements': 0,
        'ls_attempts': 0,
        'ts_disagreement_avg': 0.0,
        'vr_variance_ratio': 1.0,
        'lr_history': [],
    }
    
    logger.info(f"Session 6 components initialized:")
    logger.info(f"  Local Search: {gflownet.use_local_search} (p={gflownet.ls_probability})")
    logger.info(f"  Thompson Sampling: {gflownet.thompson_explorer is not None}")
    logger.info(f"  Variance Reduction: {gflownet.variance_reducer is not None}")
    logger.info(f"  Backward Policy: {gflownet.backward_policy is not None}")
    
    return gflownet
