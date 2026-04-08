"""
Pipeline Configuration — Central Parameter Registry
=====================================================
All hyperparameters in one place for reproducibility and paper reporting.
Every parameter is documented and justified.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List
import json
import os


@dataclass
class DataConfig:
    """Dataset generation and preprocessing parameters."""
    # Dataset sizes
    dataset_size: int = 8000              # Total molecules in dataset
    train_ratio: float = 0.80             # 80% train
    val_ratio: float = 0.10               # 10% validation
    test_ratio: float = 0.10              # 10% test
    seed: int = 42
    
    # Feature dimensions (fixed by atom/bond featurization)
    atom_feature_dim: int = 38            # 12 atom types + 6 degree + 5 charge + 5 hybrid + 1 aromatic + 1 ring + 5 H-count + 1 chirality_possible + 2 chirality_tag
    bond_feature_dim: int = 7             # 4 bond types + conjugated + ring + stereo
    
    # Biodegradable polymer monomers in dataset
    num_biodegradable_templates: int = 25
    num_non_biodegradable_templates: int = 15
    num_augmented_molecules: int = 50


@dataclass
class SurrogateConfig:
    """Surrogate model architecture and training parameters."""
    # Architecture (shared by S_bio and S_mech)
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    
    # Training
    epochs: int = 300                     # Session 8: extended for maximum convergence
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_patience: int = 8                  # ReduceLROnPlateau patience
    lr_factor: float = 0.5
    early_stopping_patience: int = 20
    carbon_budget_kg: float = 2.0
    grad_clip_norm: float = 1.0
    
    # Data subset for demo (set to None for full dataset)
    max_train_samples: int = 3000
    max_val_samples: int = 400


@dataclass
class GFlowNetConfig:
    """GFlowNet architecture, training, and generation parameters."""
    # Policy network architecture
    policy_hidden_dim: int = 256
    policy_num_layers: int = 5
    policy_dropout: float = 0.1
    
    # Molecule construction
    max_atoms: int = 30                   # Max atoms per molecule
    min_atoms: int = 10                   # Min atoms before stop is allowed (real monomers are 10-20+)
    
    # Training
    training_steps: int = 8000            # Session 8: 3× baseline for comprehensive exploration
    batch_size: int = 16                  # Session 8: reduced from 32 to prevent OOM on macOS
    policy_lr: float = 5e-4
    log_z_lr: float = 5e-3
    
    # Exploration
    temperature_init: float = 2.0         # Higher → more exploration early
    temperature_decay: float = 0.9985
    min_temperature: float = 0.5
    epsilon_init: float = 0.20            # ε-greedy exploration
    epsilon_decay: float = 0.9995
    min_epsilon: float = 0.02
    
    # Reward function — multiplicative formulation
    # R_base = S_bio^0.35 × S_mech^0.50 × S_syn^0.15
    # S_mech gets highest exponent because it's the hardest objective
    # Hard thresholds: S_bio ≥ 0.6, S_mech ≥ 0.5, S_syn ≥ 0.5
    alpha_bio: float = 1.5               # Weight biodegradability highest (legacy, used in logging)
    alpha_mech: float = 1.0              # (legacy, retained for compatibility)
    alpha_syn: float = 0.8               # (legacy, retained for compatibility)
    reward_exponent: float = 1.2          # (legacy, multiplicative formula uses fixed exponents 0.4/0.4/0.2)
    reward_min: float = 1e-6
    log_reward_min: float = -14.0
    
    # Reward shaping — bonus for desirable functional groups
    ester_bonus: float = 0.30             # Per ester bond (key for biodegradation)
    amide_bonus: float = 0.25             # Per amide bond (enzymatically cleavable)
    hydroxyl_bonus: float = 0.08          # Per hydroxyl group (hydrophilicity)
    size_bonus_threshold: int = 8         # Atoms needed for size bonus
    size_bonus: float = 0.15              # Progressive bonus for reaching threshold
    halogen_penalty: float = 0.8          # Per halogen atom (very strong — halogens blocked in action mask too)
    max_shaping_bonus: float = 0.8        # Cap on total positive shaping bonus
    
    # Trajectory Balance loss
    tb_lambda: float = 1.0               # Weight of TB loss
    
    # Generation (at inference)
    num_candidates: int = 5000            # Session 8: 5× candidates for maximum selection quality
    max_generation_attempts: int = 20000


@dataclass
class ActiveLearningConfig:
    """Active learning loop parameters."""
    num_rounds: int = 12                  # Session 8: extended AL for better surrogate accuracy
    candidates_per_round: int = 1000      # Session 8: larger candidate pool
    top_k_for_simulation: int = 80        # Session 8: more simulation data
    retrain_steps: int = 300              # Session 8: thorough recovery between rounds
    retrain_lr: float = 3e-4
    retrain_batch_size: int = 8


@dataclass
class DiscoveryConfig:
    """End-to-end discovery pipeline parameters."""
    num_candidates: int = 1000
    top_k: int = 20
    min_biodeg_improvement: float = 2.0
    similarity_weight: float = 0.3
    md_noise_level: float = 0.05


@dataclass
class PipelineConfig:
    """Master configuration for the full pipeline."""
    data: DataConfig = field(default_factory=DataConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    gflownet: GFlowNetConfig = field(default_factory=GFlowNetConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    
    # Global settings
    device: str = 'cpu'                  # Session 8: CPU prevents MPS OOM during GFlowNet training
    seed: int = 42
    checkpoint_dir: str = './checkpoints'
    results_dir: str = './results'
    carbon_log_dir: str = './carbon_logs'
    log_level: str = 'INFO'
    
    def resolve_device(self) -> str:
        """Auto-detect best available device."""
        if self.device != 'auto':
            return self.device
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        with open(path) as f:
            d = json.load(f)
        config = cls()
        config.data = DataConfig(**d.get('data', {}))
        config.surrogate = SurrogateConfig(**d.get('surrogate', {}))
        config.gflownet = GFlowNetConfig(**d.get('gflownet', {}))
        config.active_learning = ActiveLearningConfig(**d.get('active_learning', {}))
        config.discovery = DiscoveryConfig(**d.get('discovery', {}))
        for k in ['device', 'seed', 'checkpoint_dir', 'results_dir', 'carbon_log_dir', 'log_level']:
            if k in d:
                setattr(config, k, d[k])
        return config
    
    def print_summary(self):
        """Print all parameters in a structured way for the research paper."""
        lines = []
        lines.append("═" * 70)
        lines.append("  PIPELINE CONFIGURATION — PARAMETER REGISTRY")
        lines.append("═" * 70)
        
        sections = [
            ("DATA PARAMETERS", self.data),
            ("SURROGATE MODEL PARAMETERS", self.surrogate),
            ("GFLOWNET PARAMETERS", self.gflownet),
            ("ACTIVE LEARNING PARAMETERS", self.active_learning),
            ("DISCOVERY PARAMETERS", self.discovery),
        ]
        
        for section_name, section in sections:
            lines.append(f"\n  ┌─ {section_name}")
            for key, val in asdict(section).items():
                lines.append(f"  │  {key:<35} = {val}")
            lines.append(f"  └{'─'*50}")
        
        lines.append(f"\n  Device:         {self.device}")
        lines.append(f"  Seed:           {self.seed}")
        lines.append(f"  Checkpoint Dir: {self.checkpoint_dir}")
        lines.append(f"  Results Dir:    {self.results_dir}")
        lines.append("═" * 70)
        
        return "\n".join(lines)
