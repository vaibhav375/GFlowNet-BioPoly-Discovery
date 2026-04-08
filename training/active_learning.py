"""
Active Learning Loop
=====================
Implements the HPC-GFlowNet active learning cycle:

    1. GFlowNet generates candidate molecules
    2. Top candidates selected by reward
    3. MD simulation validates properties
    4. Simulation results update surrogate models
    5. GFlowNet retrains with improved surrogates
    6. Repeat

This is the core feedback loop that makes the system progressively
smarter with each iteration.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gflownet import GFlowNet, create_gflownet
from models.surrogate_bio import create_bio_model
from models.surrogate_mech import create_mech_model
from simulation.md_simulation import MDSimulator, MDSimulationResult
from evaluation.metrics import compute_all_metrics, format_metrics_table
from evaluation.green_ai_metrics import GreenAITracker, GreenAIReport

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActiveLearningLoop:
    """
    Active Learning loop that connects GFlowNet with HPC simulation.
    
    The virtuous cycle:
        Generate → Filter → Simulate → Update → Repeat
    
    Each round:
        1. Generate N candidate molecules
        2. Select top-K by reward score
        3. Run MD simulation on top-K
        4. Use simulation results to retrain surrogate models
        5. GFlowNet automatically benefits from better surrogates
    """
    
    def __init__(
        self,
        gflownet: GFlowNet,
        simulator: MDSimulator,
        config: dict = None,
        device: str = 'cpu',
    ):
        self.gflownet = gflownet
        self.simulator = simulator
        self.config = config or {}
        self.device = device
        
        # Active learning parameters
        self.candidates_per_round = self.config.get('candidates_per_round', 500)
        self.top_k_per_round = self.config.get('top_k_for_simulation', 30)
        self.num_rounds = self.config.get('num_rounds', 10)
        self.gflownet_retrain_steps = self.config.get('retrain_steps', 200)
        
        # History
        self.round_history = []
        self.all_validated_molecules = []
        self.all_simulation_results = []
        
        # Carbon tracking
        self.carbon_tracker = GreenAITracker(
            hardware='cpu' if device == 'cpu' else 'mps',
            region='us_average',
        )
    
    def run(self, num_rounds: Optional[int] = None) -> Dict:
        """
        Execute the full active learning loop.
        
        Returns:
            Summary of all rounds
        """
        num_rounds = num_rounds or self.num_rounds
        
        logger.info(f"\n{'='*70}")
        logger.info("🔄 ACTIVE LEARNING LOOP")
        logger.info(f"{'='*70}")
        logger.info(f"  Rounds: {num_rounds}")
        logger.info(f"  Candidates/round: {self.candidates_per_round}")
        logger.info(f"  Simulated/round: {self.top_k_per_round}")
        logger.info(f"  GFlowNet retrain steps: {self.gflownet_retrain_steps}")
        
        self.carbon_tracker.start()
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"\n{'─'*50}")
            logger.info(f"  ROUND {round_num}/{num_rounds}")
            logger.info(f"{'─'*50}")
            
            round_results = self._execute_round(round_num)
            self.round_history.append(round_results)
            
            # Log round summary
            logger.info(f"\n  📊 Round {round_num} Summary:")
            logger.info(f"    Generated: {round_results['num_generated']}")
            logger.info(f"    Valid: {round_results['num_valid']}")
            logger.info(f"    Simulated: {round_results['num_simulated']}")
            logger.info(f"    Stable: {round_results['num_stable']}")
            logger.info(f"    Mean reward: {round_results['mean_reward']:.4f}")
            logger.info(f"    Diversity: {round_results['diversity']:.4f}")
            logger.info(f"    Best biodeg time: {round_results.get('best_biodeg_months', 'N/A')}")
        
        # Stop carbon tracking
        total_emissions = self.carbon_tracker.stop()
        
        # Final summary
        summary = self._generate_summary(total_emissions)
        
        logger.info(f"\n{'='*70}")
        logger.info("🏁 ACTIVE LEARNING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"  Total rounds: {num_rounds}")
        logger.info(f"  Total molecules generated: {summary['total_generated']}")
        logger.info(f"  Total simulated: {summary['total_simulated']}")
        logger.info(f"  Total stable candidates: {summary['total_stable']}")
        logger.info(f"  Final diversity: {summary['final_diversity']:.4f}")
        logger.info(f"  CO₂ emissions: {total_emissions:.6f} kg")
        
        return summary
    
    def _execute_round(self, round_num: int) -> Dict:
        """Execute a single active learning round."""
        round_start = time.time()
        
        # --- Step 1: Generate candidates ---
        logger.info(f"  📦 Generating {self.candidates_per_round} candidates...")
        candidates = self.gflownet.generate_molecules(
            num_molecules=self.candidates_per_round,
            unique=True,
        )
        
        valid_candidates = [c for c in candidates if c.get('valid', False)]
        logger.info(f"    → {len(valid_candidates)} valid candidates")
        
        # --- Step 2: Select top-K using UCB acquisition ---
        # UCB = reward + β × uncertainty (from MC Dropout)
        # This selects molecules where model is uncertain → more informative
        beta = max(0.3, 1.5 - round_num * 0.15)  # Decay exploration over rounds
        
        for c in valid_candidates:
            try:
                from data.preprocessing import smiles_to_graph
                import torch
                graph = smiles_to_graph(c.get('smiles', ''))
                if graph is not None:
                    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                    # MC Dropout uncertainty for bio model
                    bio_model = self.gflownet.reward_fn.bio_model
                    if hasattr(bio_model, 'predict_with_uncertainty'):
                        _, bio_std = bio_model.predict_with_uncertainty(graph, K=5)
                    else:
                        bio_std = 0.0
                    c['uncertainty'] = float(bio_std)
                else:
                    c['uncertainty'] = 0.0
            except Exception:
                c['uncertainty'] = 0.0
            
            c['ucb_score'] = c.get('reward', 0) + beta * c.get('uncertainty', 0)
        
        sorted_candidates = sorted(valid_candidates, key=lambda x: x.get('ucb_score', 0), reverse=True)
        top_k = sorted_candidates[:self.top_k_per_round]
        
        avg_uncertainty = np.mean([c.get('uncertainty', 0) for c in top_k]) if top_k else 0
        logger.info(f"  🏆 Top-{self.top_k_per_round} by UCB (β={beta:.2f}, "
                    f"avg_uncertainty={avg_uncertainty:.4f})")
        
        # --- Step 3: Simulate ---
        logger.info(f"  🔬 Running MD simulation on {len(top_k)} molecules...")
        sim_smiles = [c['smiles'] for c in top_k]
        sim_results = self.simulator.simulate_batch(sim_smiles)
        
        # Analyze simulation results
        stable_results = [r for r in sim_results if r.is_stable]
        unstable_results = [r for r in sim_results if not r.is_stable]
        
        logger.info(f"    → Stable: {len(stable_results)}/{len(sim_results)}")
        if unstable_results:
            logger.info(f"    → Unstable: {len(unstable_results)} "
                       f"(reasons: {', '.join(set(w for r in unstable_results for w in r.warnings))})")
        
        # Store validated molecules
        self.all_simulation_results.extend(sim_results)
        for r in stable_results:
            self.all_validated_molecules.append(r.to_dict())
        
        # --- Step 4: Update surrogate models ---
        if len(stable_results) > 0:
            logger.info(f"  🧠 Updating surrogate models with {len(sim_results)} simulation results...")
            self._update_surrogates(sim_results)
        
        # --- Step 5: Retrain GFlowNet ---
        logger.info(f"  🔧 Retraining GFlowNet ({self.gflownet_retrain_steps} steps)...")
        optimizer = torch.optim.Adam(self.gflownet.parameters(), lr=0.0003)
        
        retrain_losses = []
        for _ in range(self.gflownet_retrain_steps):
            metrics = self.gflownet.train_step(optimizer, batch_size=8)
            retrain_losses.append(metrics['loss'])
        
        logger.info(f"    → Retrain loss: {np.mean(retrain_losses):.4f}")
        
        # Compute round metrics
        eval_metrics = compute_all_metrics(candidates)
        
        # Best biodegradation time from simulation
        best_biodeg = min(
            (r.predicted_biodeg_rate for r in stable_results),
            default=999.0
        )
        
        round_time = time.time() - round_start
        
        return {
            'round': round_num,
            'num_generated': len(candidates),
            'num_valid': len(valid_candidates),
            'num_simulated': len(sim_results),
            'num_stable': len(stable_results),
            'mean_reward': eval_metrics.get('mean_reward', 0),
            'diversity': eval_metrics.get('diversity', 0),
            'validity_rate': eval_metrics.get('validity_rate', 0),
            'best_biodeg_months': best_biodeg,
            'retrain_loss': np.mean(retrain_losses),
            'round_time_seconds': round_time,
        }
    
    def _update_surrogates(self, sim_results: List[MDSimulationResult]):
        """
        Fine-tune surrogate models with MD simulation results.
        
        Uses simulation data as new training examples to close the
        active learning loop. Each simulation result provides ground-truth
        biodegradation rate and mechanical properties that improve
        the surrogate predictions.
        """
        from data.preprocessing import smiles_to_graph, compute_synthetic_biodegradability_label
        
        # Collect training data from simulation results
        bio_data = []
        mech_data = []
        
        for result in sim_results:
            graph = smiles_to_graph(result.smiles)
            if graph is None:
                continue
            
            # Biodegradability target: convert months → [0,1] score
            # Faster degradation = higher score
            # 1 month → ~0.95, 6 months → ~0.75, 24 months → ~0.4, 120+ → ~0.1
            biodeg_score = max(0.0, min(1.0, 1.0 - (result.predicted_biodeg_rate / 150.0)))
            
            bio_graph = graph.clone()
            bio_graph.y = torch.tensor([biodeg_score], dtype=torch.float)
            bio_data.append(bio_graph)
            
            # Mechanical properties target: normalize
            mech_graph = graph.clone()
            # Tensile: normalize to roughly [0, 1] by dividing by 100 MPa
            mech_graph.y = torch.tensor(
                [result.predicted_tensile / 100.0],
                dtype=torch.float,
            )
            mech_data.append(mech_graph)
        
        if len(bio_data) < 3:
            logger.info(f"    → Too few valid samples ({len(bio_data)}) for surrogate update")
            return
        
        # Fine-tune S_bio
        if hasattr(self.gflownet, 'reward_fn') and hasattr(self.gflownet.reward_fn, 'bio_model'):
            bio_model = self.gflownet.reward_fn.bio_model
            bio_model.train()
            optimizer = torch.optim.Adam(bio_model.parameters(), lr=1e-4)
            
            # Mini-batch fine-tuning (3 epochs over simulation data)
            for epoch in range(3):
                total_loss = 0.0
                np.random.shuffle(bio_data)
                for g in bio_data:
                    g = g.to(self.device)
                    pred = bio_model(g)
                    loss = torch.nn.functional.mse_loss(pred.view(-1), g.y.view(-1).to(self.device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            bio_model.eval()
            avg_loss = total_loss / max(len(bio_data), 1)
            logger.info(f"    → S_bio fine-tuned on {len(bio_data)} samples (loss: {avg_loss:.4f})")
        
        # Fine-tune S_mech
        if hasattr(self.gflownet, 'reward_fn') and hasattr(self.gflownet.reward_fn, 'mech_model'):
            mech_model = self.gflownet.reward_fn.mech_model
            mech_model.train()
            optimizer = torch.optim.Adam(mech_model.parameters(), lr=1e-4)
            
            for epoch in range(3):
                total_loss = 0.0
                np.random.shuffle(mech_data)
                for g in mech_data:
                    g = g.to(self.device)
                    out = mech_model(g)
                    # Use 'tensile' prediction head; target is tensile/100
                    pred = out['tensile'] if isinstance(out, dict) else out.view(-1)
                    loss = torch.nn.functional.mse_loss(pred.view(-1), g.y.view(-1).to(self.device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            mech_model.eval()
            avg_loss = total_loss / max(len(mech_data), 1)
            logger.info(f"    → S_mech fine-tuned on {len(mech_data)} samples (loss: {avg_loss:.4f})")
        
        logger.info(f"    → Surrogate update complete ({len(bio_data)} simulation data points)")
    
    def _generate_summary(self, total_emissions: float) -> Dict:
        """Generate comprehensive summary of all rounds."""
        all_generated = sum(r['num_generated'] for r in self.round_history)
        all_simulated = sum(r['num_simulated'] for r in self.round_history)
        all_stable = sum(r['num_stable'] for r in self.round_history)
        
        # Progression metrics
        rewards_over_rounds = [r['mean_reward'] for r in self.round_history]
        diversity_over_rounds = [r['diversity'] for r in self.round_history]
        
        summary = {
            'total_rounds': len(self.round_history),
            'total_generated': all_generated,
            'total_simulated': all_simulated,
            'total_stable': all_stable,
            'total_validated_molecules': len(self.all_validated_molecules),
            'final_diversity': diversity_over_rounds[-1] if diversity_over_rounds else 0,
            'final_mean_reward': rewards_over_rounds[-1] if rewards_over_rounds else 0,
            'reward_improvement': (
                (rewards_over_rounds[-1] - rewards_over_rounds[0]) / max(rewards_over_rounds[0], 1e-6) * 100
                if len(rewards_over_rounds) >= 2 else 0
            ),
            'total_emissions_kg': total_emissions,
            'emissions_per_molecule': total_emissions / max(all_generated, 1),
            'round_details': self.round_history,
            'validated_molecules': self.all_validated_molecules[:50],  # Top 50
        }
        
        return summary
    
    def save_results(self, output_dir: str):
        """Save all active learning results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save round history
        with open(os.path.join(output_dir, 'active_learning_history.json'), 'w') as f:
            json.dump(self.round_history, f, indent=2, default=str)
        
        # Save validated molecules
        with open(os.path.join(output_dir, 'validated_molecules.json'), 'w') as f:
            json.dump(self.all_validated_molecules, f, indent=2, default=str)
        
        logger.info(f"Active learning results saved to {output_dir}")


def main():
    """Run the active learning loop."""
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    config = {
        'candidates_per_round': 200,
        'top_k_for_simulation': 20,
        'num_rounds': 5,
        'retrain_steps': 100,
        'max_atoms': 25,
        'temperature': 1.5,
        'epsilon': 0.15,
        'policy_hidden_dim': 128,
        'policy_num_layers': 3,
    }
    
    # Create GFlowNet
    gflownet = create_gflownet(config, device=device)
    
    # Load pre-trained checkpoints if available
    ckpt_path = './checkpoints/gflownet_best.pt'
    if os.path.exists(ckpt_path):
        gflownet.load(ckpt_path)
        logger.info("Loaded pre-trained GFlowNet")
    
    # Create simulator
    simulator = MDSimulator(noise_level=0.1)
    
    # Run active learning
    al_loop = ActiveLearningLoop(
        gflownet=gflownet,
        simulator=simulator,
        config=config,
        device=device,
    )
    
    summary = al_loop.run(num_rounds=config['num_rounds'])
    al_loop.save_results('./results/active_learning')
    
    return summary


if __name__ == "__main__":
    main()
