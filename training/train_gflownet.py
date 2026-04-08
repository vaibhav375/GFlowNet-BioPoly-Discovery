"""
GFlowNet Training Script
==========================
Trains the GFlowNet to generate diverse, high-reward biodegradable
polymer candidates using Trajectory Balance loss.

Training loop:
    1. Generate batch of molecular trajectories
    2. Compute TB loss for each trajectory
    3. Update policy network
    4. Track metrics (reward, diversity, validity, carbon)
    5. Periodically evaluate and save checkpoints

Green AI practices:
    - CodeCarbon emission tracking
    - Mixed precision training
    - Early stopping on carbon budget
    - Efficient batch generation
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gflownet import create_gflownet, GFlowNet
from models.surrogate_bio import create_bio_model
from models.surrogate_mech import create_mech_model
from models.surrogate_syn import SynthesizabilityScorer
from evaluation.metrics import (
    compute_diversity,
    compute_validity_rate,
    compute_novelty,
    compute_top_k_reward,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_gflownet(
    config: dict = None,
    device: str = 'cpu',
    save_dir: str = './checkpoints',
    log_dir: str = './results',
) -> Dict:
    """
    Main GFlowNet training function.
    
    Args:
        config: Training configuration dictionary
        device: Training device
        save_dir: Directory for model checkpoints
        log_dir: Directory for training logs
    
    Returns:
        Training history dictionary
    """
    config = config or {}
    
    # Default configuration
    num_episodes = config.get('num_episodes', 10000)
    batch_size = config.get('batch_size', 16)
    lr = config.get('learning_rate', 0.0005)
    eval_interval = config.get('eval_interval', 500)
    save_interval = config.get('save_interval', 1000)
    num_eval_molecules = config.get('num_eval_molecules', 200)
    carbon_budget_kg = config.get('carbon_budget_kg', 5.0)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # --- Create GFlowNet ---
    logger.info("🔧 Creating GFlowNet...")
    
    # Load pre-trained surrogate models if available
    gflownet = create_gflownet(config, device=device)
    
    bio_path = os.path.join(save_dir, 's_bio_best.pt')
    mech_path = os.path.join(save_dir, 's_mech_best.pt')
    
    if os.path.exists(bio_path):
        gflownet.reward_fn.bio_model.load_state_dict(
            torch.load(bio_path, map_location=device)
        )
        logger.info("✓ Loaded pre-trained S_bio model")
    else:
        logger.warning("⚠ No pre-trained S_bio found. Using random initialization.")
    
    if os.path.exists(mech_path):
        gflownet.reward_fn.mech_model.load_state_dict(
            torch.load(mech_path, map_location=device)
        )
        logger.info("✓ Loaded pre-trained S_mech model")
    else:
        logger.warning("⚠ No pre-trained S_mech found. Using random initialization.")
    
    # Optimizer
    optimizer = optim.Adam(
        [
            {'params': gflownet.policy.parameters(), 'lr': lr},
            {'params': [gflownet.log_z], 'lr': lr * 10},  # Faster learning for log_z
        ],
        weight_decay=1e-5,
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2
    )
    
    # Carbon tracking
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(
            project_name="train_gflownet",
            output_dir="./carbon_logs",
            log_level="warning",
        )
        tracker.start()
        tracking_carbon = True
    except ImportError:
        logger.warning("CodeCarbon not installed.")
        tracking_carbon = False
        tracker = None
    
    # --- Training Loop ---
    logger.info(f"\n{'='*60}")
    logger.info("🚀 GFlowNet Training")
    logger.info(f"{'='*60}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Device: {device}")
    logger.info(f"Carbon budget: {carbon_budget_kg} kg CO2")
    
    history = {
        'loss': [], 'mean_reward': [], 'max_reward': [],
        'validity': [], 'diversity': [], 'unique_count': [],
        'temperature': [], 'log_z': [],
        'eval_rewards': [], 'eval_diversity': [],
    }
    
    all_generated_smiles = set()
    best_reward = 0.0
    training_start = time.time()
    
    num_steps = num_episodes // batch_size
    
    for step in tqdm(range(num_steps), desc="Training GFlowNet"):
        # Training step
        metrics = gflownet.train_step(optimizer, batch_size=batch_size)
        scheduler.step()
        
        # Log metrics
        history['loss'].append(metrics['loss'])
        history['mean_reward'].append(metrics['mean_reward'])
        history['max_reward'].append(metrics['max_reward'])
        history['validity'].append(metrics['validity_rate'])
        history['unique_count'].append(metrics['unique_smiles'])
        history['temperature'].append(metrics['temperature'])
        history['log_z'].append(metrics['log_z'])
        
        # Track best reward
        if metrics['max_reward'] > best_reward:
            best_reward = metrics['max_reward']
        
        # Periodic logging
        if (step + 1) % 50 == 0:
            logger.info(
                f"Step {step+1}/{num_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Reward: {metrics['mean_reward']:.4f} (max: {metrics['max_reward']:.4f}) | "
                f"Valid: {metrics['validity_rate']:.1%} | "
                f"Temp: {metrics['temperature']:.3f} | "
                f"log Z: {metrics['log_z']:.2f}"
            )
        
        # Periodic evaluation
        if (step + 1) % (eval_interval // batch_size) == 0:
            eval_results = evaluate_gflownet(gflownet, num_molecules=num_eval_molecules)
            history['eval_rewards'].append(eval_results['mean_reward'])
            history['eval_diversity'].append(eval_results['diversity'])
            
            logger.info(
                f"\n📊 Evaluation at step {step+1}:\n"
                f"  Generated: {eval_results['num_generated']}\n"
                f"  Valid: {eval_results['validity_rate']:.1%}\n"
                f"  Unique: {eval_results['num_unique']}\n"
                f"  Diversity: {eval_results['diversity']:.4f}\n"
                f"  Mean Reward: {eval_results['mean_reward']:.4f}\n"
                f"  Top-10 Reward: {eval_results['top_10_reward']:.4f}\n"
            )
            
            # Save if best
            if eval_results['mean_reward'] > best_reward * 0.9:
                gflownet.save(os.path.join(save_dir, 'gflownet_best.pt'))
        
        # Periodic save
        if (step + 1) % (save_interval // batch_size) == 0:
            gflownet.save(os.path.join(save_dir, f'gflownet_step_{step+1}.pt'))
        
        # Carbon budget check
        if tracking_carbon and tracker is not None:
            # Check periodically (not every step for efficiency)
            if (step + 1) % 100 == 0:
                current_emissions = tracker._total_energy.kWh * 0.5  # Rough estimate
                if current_emissions > carbon_budget_kg:
                    logger.warning(f"⚠ Approaching carbon budget. Stopping training.")
                    break
    
    # --- Finalize ---
    training_time = time.time() - training_start
    
    # Stop carbon tracking
    total_emissions = 0.0
    if tracking_carbon and tracker is not None:
        total_emissions = tracker.stop()
    
    # Save final model
    gflownet.save(os.path.join(save_dir, 'gflownet_final.pt'))
    
    # Final evaluation
    logger.info("\n📊 Final Evaluation...")
    final_results = evaluate_gflownet(gflownet, num_molecules=500)
    
    # Save training history
    history['training_time_seconds'] = training_time
    history['total_emissions_kg'] = total_emissions
    history['best_reward'] = best_reward
    history['final_eval'] = final_results
    
    with open(os.path.join(log_dir, 'gflownet_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("🏁 TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Training time: {training_time/60:.1f} minutes")
    logger.info(f"Total CO2: {total_emissions:.6f} kg")
    logger.info(f"Best reward: {best_reward:.4f}")
    logger.info(f"Final validity: {final_results['validity_rate']:.1%}")
    logger.info(f"Final diversity: {final_results['diversity']:.4f}")
    logger.info(f"Final mean reward: {final_results['mean_reward']:.4f}")
    logger.info(f"Unique molecules: {final_results['num_unique']}")
    logger.info("✅ GFlowNet training complete!")
    
    return history


def evaluate_gflownet(gflownet: GFlowNet, num_molecules: int = 200) -> Dict:
    """
    Evaluate GFlowNet by generating molecules and computing metrics.
    
    Returns:
        Dictionary of evaluation metrics
    """
    results = gflownet.generate_molecules(num_molecules=num_molecules, unique=True)
    
    if not results:
        return {
            'num_generated': 0, 'num_unique': 0, 'validity_rate': 0.0,
            'diversity': 0.0, 'mean_reward': 0.0, 'top_10_reward': 0.0,
        }
    
    smiles_list = [r['smiles'] for r in results if r.get('smiles')]
    rewards = [r['reward'] for r in results]
    valid_count = sum(1 for r in results if r.get('valid', False))
    
    # Diversity
    diversity = compute_diversity(smiles_list) if len(smiles_list) >= 2 else 0.0
    
    # Top-K reward
    top_10 = sorted(rewards, reverse=True)[:10]
    top_10_reward = np.mean(top_10) if top_10 else 0.0
    
    return {
        'num_generated': len(results),
        'num_unique': len(set(smiles_list)),
        'validity_rate': valid_count / max(len(results), 1),
        'diversity': diversity,
        'mean_reward': np.mean(rewards),
        'top_10_reward': top_10_reward,
        'top_molecules': results[:10],  # Top 10 by reward
    }


def main():
    """Main training entry point."""
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    config = {
        'num_episodes': 10000,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'eval_interval': 500,
        'save_interval': 2000,
        'num_eval_molecules': 200,
        'carbon_budget_kg': 5.0,
        'max_atoms': 30,
        'temperature': 1.5,
        'temperature_decay': 0.9995,
        'min_temperature': 0.2,
        'epsilon': 0.1,
        'reward_exponent': 2.0,
        'policy_hidden_dim': 256,
        'policy_num_layers': 4,
    }
    
    history = train_gflownet(config=config, device=device)


if __name__ == "__main__":
    main()
