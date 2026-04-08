"""
Ablation Study — Publication Improvement #8
=============================================
Systematically evaluates the contribution of each improvement to
overall pipeline performance. Designed for Table 2 in the paper.

Usage:
    python run_ablation.py          # Full ablation (slow)
    python run_ablation.py --quick  # Quick 50-step test
"""

import sys
import time
import json
import os
import numpy as np
import torch
from typing import Dict, List

# Import pipeline components
from data.preprocessing import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from data.real_polymer_data import get_all_real_data
from models.gflownet import GFlowNet, ReplayBuffer, RewardFunction
from models.surrogate_bio import BiodegradabilityPredictor
from models.surrogate_mech import MechanicalPropertiesPredictor
from models.surrogate_syn import SynthesizabilityScorer
from models.policy_network import PolicyNetwork


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

ABLATION_CONFIGS = {
    'baseline': {
        'description': 'Trajectory Balance only (no improvements)',
        'use_subtb': False,
        'replay_capacity': 0,
        'mogfn_enabled': False,
        'diversity_threshold': 1.0,
    },
    '+subtb': {
        'description': 'Baseline + Sub-Trajectory Balance loss',
        'use_subtb': True,
        'subtb_weight': 0.5,
        'replay_capacity': 0,
        'mogfn_enabled': False,
        'diversity_threshold': 1.0,
    },
    '+replay': {
        'description': 'Baseline + Replay Buffer',
        'use_subtb': False,
        'replay_capacity': 2000,
        'replay_alpha': 0.6,
        'mogfn_enabled': False,
        'diversity_threshold': 1.0,
    },
    '+subtb+replay': {
        'description': 'SubTB + Replay Buffer',
        'use_subtb': True,
        'subtb_weight': 0.5,
        'replay_capacity': 2000,
        'replay_alpha': 0.6,
        'mogfn_enabled': False,
        'diversity_threshold': 1.0,
    },
    '+diversity': {
        'description': 'SubTB + Replay + Diversity Penalty',
        'use_subtb': True,
        'subtb_weight': 0.5,
        'replay_capacity': 2000,
        'replay_alpha': 0.6,
        'mogfn_enabled': False,
        'diversity_threshold': 0.85,
    },
    'full': {
        'description': 'All improvements (SubTB + Replay + Diversity + MOGFN)',
        'use_subtb': True,
        'subtb_weight': 0.5,
        'replay_capacity': 2000,
        'replay_alpha': 0.6,
        'mogfn_enabled': True,
        'diversity_threshold': 0.85,
    },
}


def create_gflownet_for_ablation(config: Dict, device='cpu') -> tuple:
    """Create a properly initialized GFlowNet with surrogates for ablation."""
    # Build surrogate models
    bio_model = BiodegradabilityPredictor(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=128, num_layers=4, dropout=0.2,
    )
    mech_model = MechanicalPropertiesPredictor(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=128, num_layers=4, dropout=0.2,
    )
    syn_scorer = SynthesizabilityScorer()

    # Load trained checkpoints if available
    bio_path = './checkpoints/s_bio_best.pt'
    mech_path = './checkpoints/s_mech_best.pt'
    if os.path.exists(bio_path):
        state = torch.load(bio_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            bio_model.load_state_dict(state['model_state_dict'])
        else:
            bio_model.load_state_dict(state)
    if os.path.exists(mech_path):
        state = torch.load(mech_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            mech_model.load_state_dict(state['model_state_dict'])
        else:
            mech_model.load_state_dict(state)

    # Build RewardFunction
    reward_fn = RewardFunction(
        bio_model=bio_model, mech_model=mech_model, syn_scorer=syn_scorer,
        alpha_bio=1.5, alpha_mech=1.0, alpha_syn=0.8,
        reward_exponent=1.2, reward_min=1e-6, device=device,
        ester_bonus=0.30, amide_bonus=0.25, hydroxyl_bonus=0.08,
        size_bonus_threshold=8, size_bonus=0.15,
        halogen_penalty=0.8, max_shaping_bonus=0.8,
    )

    # Build PolicyNetwork
    policy = PolicyNetwork(
        state_dim=ATOM_FEATURE_DIM, hidden_dim=256,
        num_layers=5, num_actions=18, dropout=0.1,
    )

    # Build GFlowNet with ablation config
    gfn = GFlowNet(
        policy_network=policy, reward_function=reward_fn,
        config=config, device=device,
    )
    return gfn


def run_ablation_experiment(
    config_name: str,
    config: Dict,
    num_steps: int = 200,
    batch_size: int = 16,
    seed: int = 42,
) -> Dict:
    """Run a single ablation experiment and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  ABLATION: {config_name}")
    print(f"  {config.get('description', '')}")
    print(f"{'='*60}")

    gfn = create_gflownet_for_ablation(config)
    optimizer = torch.optim.Adam(gfn.parameters(), lr=1e-3)

    metrics_acc = {
        'config_name': config_name,
        'description': config.get('description', ''),
        'losses': [],
        'rewards': [],
        'valid_rates': [],
        'unique_smiles': set(),
        'best_reward': 0.0,
        'pareto_size': 0,
    }

    start_time = time.time()

    for step in range(num_steps):
        result = gfn.train_step(optimizer, batch_size=batch_size)

        # Correct metric keys from train_step return dict
        metrics_acc['losses'].append(result.get('loss', 0.0))
        metrics_acc['rewards'].append(result.get('mean_reward', 0.0))
        metrics_acc['valid_rates'].append(result.get('validity_rate', 0.0))

        if result.get('max_reward', 0) > metrics_acc['best_reward']:
            metrics_acc['best_reward'] = result['max_reward']

        # Track unique molecules from num_generated
        metrics_acc['unique_smiles'].add(str(result.get('unique_smiles', 0)))

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{num_steps}: "
                  f"loss={result.get('loss', 0):.4f}, "
                  f"reward={result.get('mean_reward', 0):.4f}, "
                  f"valid={result.get('validity_rate', 0):.1%}, "
                  f"unique={result.get('unique_smiles', 0)}")

    elapsed = time.time() - start_time

    if hasattr(gfn, 'pareto_front') and gfn.pareto_front is not None:
        metrics_acc['pareto_size'] = gfn.pareto_front.size

    # Final summary metrics (last 20 steps average)
    last_n = min(20, len(metrics_acc['losses']))
    metrics_acc['elapsed_seconds'] = elapsed
    metrics_acc['num_unique'] = int(sum(float(x) for x in metrics_acc['unique_smiles'] if x.replace('.','').isdigit()))
    metrics_acc['final_loss'] = float(np.mean(metrics_acc['losses'][-last_n:])) if metrics_acc['losses'] else 0
    metrics_acc['final_reward'] = float(np.mean(metrics_acc['rewards'][-last_n:])) if metrics_acc['rewards'] else 0
    metrics_acc['final_valid_rate'] = float(np.mean(metrics_acc['valid_rates'][-last_n:])) if metrics_acc['valid_rates'] else 0
    metrics_acc['replay_buffer_size'] = len(gfn.replay_buffer)
    metrics_acc['best_reward'] = float(metrics_acc['best_reward'])

    # Clean up non-serializable
    metrics_acc['unique_smiles'] = []

    print(f"\n  RESULTS: loss={metrics_acc['final_loss']:.4f}, "
          f"reward={metrics_acc['final_reward']:.4f}, "
          f"valid={metrics_acc['final_valid_rate']:.1%}, "
          f"best={metrics_acc['best_reward']:.4f}, "
          f"time={elapsed:.1f}s")

    return metrics_acc


def run_full_ablation(num_steps: int = 200, batch_size: int = 16):
    """Run complete ablation study."""
    print("=" * 70)
    print("  ABLATION STUDY — GFlowNet BioPoly Discovery")
    print("  Publication Improvement #8")
    print("=" * 70)
    print(f"  Steps per config: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Configurations: {len(ABLATION_CONFIGS)}")

    results = {}
    for name, config in ABLATION_CONFIGS.items():
        results[name] = run_ablation_experiment(
            name, config, num_steps=num_steps, batch_size=batch_size
        )

    # Summary table
    print("\n" + "=" * 85)
    print("  ABLATION SUMMARY")
    print("=" * 85)
    print(f"{'Config':<20} {'Loss':>8} {'Mean R':>8} {'Best R':>8} {'Valid%':>8} {'Replay':>8} {'Pareto':>8} {'Time':>8}")
    print("-" * 85)
    for name, m in results.items():
        print(f"{name:<20} {m['final_loss']:>8.4f} {m['final_reward']:>8.4f} "
              f"{m['best_reward']:>8.4f} {m['final_valid_rate']:>7.1%} "
              f"{m['replay_buffer_size']:>8} {m['pareto_size']:>8} {m['elapsed_seconds']:>7.1f}s")

    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/ablation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  📄 Results saved to {output_path}")

    return results


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    num_steps = 50 if quick else 200
    batch_size = 8 if quick else 16

    if quick:
        print("  ⚡ Quick mode: 50 steps, batch_size=8")

    run_full_ablation(num_steps=num_steps, batch_size=batch_size)
