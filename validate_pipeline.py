"""
Pipeline Validation & Tuning Script
=====================================
Validates trained surrogate models and reward function against
experimentally validated polymer benchmarks.

Compares:
  - S_bio predictions vs real biodeg scores (OECD 301)
  - S_mech predictions vs real tensile/Tg values
  - S_syn (SA score) for known synthesizable monomers
  - Combined reward R for ideal candidates vs non-degradable plastics

Ideal Targets (from literature):
  - PLA monomer:  S_bio≥0.85, S_mech≥0.55, S_syn≥0.80, R≥0.65
  - PCL monomer:  S_bio≥0.75, S_mech≥0.40, S_syn≥0.85, R≥0.55
  - PBS monomer:  S_bio≥0.80, S_mech≥0.50, S_syn≥0.75, R≥0.58
  - PHB monomer:  S_bio≥0.85, S_mech≥0.45, S_syn≥0.70, R≥0.55
  - PGA monomer:  S_bio≥0.90, S_mech≥0.60, S_syn≥0.80, R≥0.68
  - PE (non-biodeg): S_bio≤0.15, R≤0.20
  - PS (non-biodeg): S_bio≤0.15, R≤0.20
  - PET (non-biodeg): S_bio≤0.20, R≤0.25
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)

import torch
import numpy as np


def load_reward_function(device='cpu'):
    """Load trained surrogates into RewardFunction."""
    from models.gflownet import RewardFunction
    from models.surrogate_bio import BiodegradabilityPredictor
    from models.surrogate_mech import MechanicalPropertiesPredictor
    from models.surrogate_syn import SynthesizabilityScorer
    from configs.pipeline_config import PipelineConfig
    from data.preprocessing import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
    
    config = PipelineConfig()
    
    # Build individual models
    bio_model = BiodegradabilityPredictor(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=config.surrogate.hidden_dim,
        num_layers=config.surrogate.num_layers,
        dropout=config.surrogate.dropout,
    )
    
    mech_model = MechanicalPropertiesPredictor(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=config.surrogate.hidden_dim,
        num_layers=config.surrogate.num_layers,
        dropout=config.surrogate.dropout,
    )
    
    syn_scorer = SynthesizabilityScorer()
    
    # Load trained checkpoints if they exist
    bio_path = './checkpoints/s_bio_best.pt'
    mech_path = './checkpoints/s_mech_best.pt'
    
    if os.path.exists(bio_path):
        state = torch.load(bio_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            bio_model.load_state_dict(state['model_state_dict'])
        else:
            bio_model.load_state_dict(state)
        print(f"✅ Loaded S_bio from {bio_path}")
    else:
        print(f"⚠️  S_bio checkpoint not found: {bio_path}")
    
    if os.path.exists(mech_path):
        state = torch.load(mech_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            mech_model.load_state_dict(state['model_state_dict'])
        else:
            mech_model.load_state_dict(state)
        print(f"✅ Loaded S_mech from {mech_path}")
    else:
        print(f"⚠️  S_mech checkpoint not found: {mech_path}")
    
    # Create RewardFunction with proper models
    gfn_cfg = config.gflownet
    rf = RewardFunction(
        bio_model=bio_model,
        mech_model=mech_model,
        syn_scorer=syn_scorer,
        alpha_bio=gfn_cfg.alpha_bio,
        alpha_mech=gfn_cfg.alpha_mech,
        alpha_syn=gfn_cfg.alpha_syn,
        reward_exponent=gfn_cfg.reward_exponent,
        reward_min=gfn_cfg.reward_min,
        device=device,
        ester_bonus=gfn_cfg.ester_bonus,
        amide_bonus=gfn_cfg.amide_bonus,
        hydroxyl_bonus=gfn_cfg.hydroxyl_bonus,
        size_bonus_threshold=gfn_cfg.size_bonus_threshold,
        size_bonus=gfn_cfg.size_bonus,
        halogen_penalty=gfn_cfg.halogen_penalty,
        max_shaping_bonus=gfn_cfg.max_shaping_bonus,
    )
    
    return rf


# ═══════════════════════════════════════════════════════
# BENCHMARK MOLECULES WITH EXPECTED IDEAL VALUES
# ═══════════════════════════════════════════════════════

BENCHMARKS = {
    # ── Biodegradable targets (should score HIGH) ──
    'L-Lactic acid (PLA)': {
        'smiles': 'CC(O)C(=O)O',
        'ideal_bio': 0.92, 'ideal_mech': 0.50, 'ideal_syn': 0.80,
        'ideal_reward_min': 0.55, 'category': 'biodegradable',
        'real_tg': 53.0, 'real_tensile': 50.0,
    },
    'Lactide (PLA dimer)': {
        'smiles': 'OC(=O)C(C)OC(=O)C(C)O',
        'ideal_bio': 0.88, 'ideal_mech': 0.55, 'ideal_syn': 0.82,
        'ideal_reward_min': 0.60, 'category': 'biodegradable',
        'real_tg': 58.0, 'real_tensile': 55.0,
    },
    'ε-Caprolactone (PCL)': {
        'smiles': 'O=C1CCCCCO1',
        'ideal_bio': 0.82, 'ideal_mech': 0.40, 'ideal_syn': 0.90,
        'ideal_reward_min': 0.50, 'category': 'biodegradable',
        'real_tg': -60.0, 'real_tensile': 25.0,
    },
    'Succinic acid (PBS)': {
        'smiles': 'OC(=O)CCC(=O)O',
        'ideal_bio': 0.88, 'ideal_mech': 0.50, 'ideal_syn': 0.78,
        'ideal_reward_min': 0.55, 'category': 'biodegradable',
        'real_tg': -32.0, 'real_tensile': 35.0,
    },
    '3-HB acid (PHB)': {
        'smiles': 'CC(O)CC(=O)O',
        'ideal_bio': 0.90, 'ideal_mech': 0.45, 'ideal_syn': 0.75,
        'ideal_reward_min': 0.50, 'category': 'biodegradable',
        'real_tg': 5.0, 'real_tensile': 40.0,
    },
    'Glycolic acid (PGA)': {
        'smiles': 'OCC(=O)O',
        'ideal_bio': 0.95, 'ideal_mech': 0.60, 'ideal_syn': 0.85,
        'ideal_reward_min': 0.65, 'category': 'biodegradable',
        'real_tg': 36.0, 'real_tensile': 60.0,
    },
    'PBS dimer': {
        'smiles': 'O=C(CCC(=O)OCCCCO)OCCCCO',
        'ideal_bio': 0.84, 'ideal_mech': 0.50, 'ideal_syn': 0.75,
        'ideal_reward_min': 0.52, 'category': 'biodegradable',
        'real_tg': -35.0, 'real_tensile': 36.0,
    },
    'Glucose (cellulose)': {
        'smiles': 'OCC1OC(O)C(O)C(O)C1O',
        'ideal_bio': 0.95, 'ideal_mech': 0.35, 'ideal_syn': 0.75,
        'ideal_reward_min': 0.45, 'category': 'biodegradable',
        'real_tg': 80.0, 'real_tensile': 15.0,
    },

    # ── Non-biodegradable (should score LOW on S_bio) ──
    'Decane (PE model)': {
        'smiles': 'CCCCCCCCCC',
        'ideal_bio': 0.05, 'ideal_mech': 0.30, 'ideal_syn': 0.90,
        'ideal_reward_min': 0.0, 'category': 'non-biodegradable',
        'real_tg': -120.0, 'real_tensile': 25.0,
    },
    'PS trimer': {
        'smiles': 'CC(c1ccccc1)CC(c1ccccc1)C',
        'ideal_bio': 0.05, 'ideal_mech': 0.50, 'ideal_syn': 0.60,
        'ideal_reward_min': 0.0, 'category': 'non-biodegradable',
        'real_tg': 100.0, 'real_tensile': 40.0,
    },
    'PET repeat unit': {
        'smiles': 'O=C(c1ccc(C(=O)OCCO)cc1)OCCO',
        'ideal_bio': 0.08, 'ideal_mech': 0.55, 'ideal_syn': 0.70,
        'ideal_reward_min': 0.0, 'category': 'non-biodegradable',
        'real_tg': 70.0, 'real_tensile': 55.0,
    },
    'PVC oligomer': {
        'smiles': 'CC(Cl)CC(Cl)CC(Cl)C',
        'ideal_bio': 0.05, 'ideal_mech': 0.50, 'ideal_syn': 0.55,
        'ideal_reward_min': 0.0, 'category': 'non-biodegradable',
        'real_tg': 80.0, 'real_tensile': 50.0,
    },
    'PP oligomer': {
        'smiles': 'CC(C)CC(C)CC(C)C',
        'ideal_bio': 0.05, 'ideal_mech': 0.40, 'ideal_syn': 0.80,
        'ideal_reward_min': 0.0, 'category': 'non-biodegradable',
        'real_tg': -10.0, 'real_tensile': 35.0,
    },
}


def validate(rf):
    """Run all benchmarks and produce a detailed report."""
    
    print("\n" + "═" * 80)
    print("  PIPELINE VALIDATION REPORT")
    print("═" * 80)
    
    results = {}
    biodeg_errors = []
    non_biodeg_errors = []
    total_reward_pass = 0
    total_checks = 0
    
    for name, bench in BENCHMARKS.items():
        r = rf.compute_reward(bench['smiles'])
        results[name] = r
        
        # Compute errors
        bio_err = abs(r['s_bio'] - bench['ideal_bio'])
        
        is_pass = True
        status_parts = []
        
        if bench['category'] == 'biodegradable':
            biodeg_errors.append(bio_err)
            if r['s_bio'] < bench['ideal_bio'] * 0.7:
                status_parts.append(f"⚠️ S_bio too low ({r['s_bio']:.3f} vs ideal {bench['ideal_bio']:.2f})")
                is_pass = False
            if r['reward'] < bench['ideal_reward_min']:
                status_parts.append(f"⚠️ Reward too low ({r['reward']:.3f} vs min {bench['ideal_reward_min']:.2f})")
                is_pass = False
        else:
            non_biodeg_errors.append(bio_err)
            if r['s_bio'] > 0.30:
                status_parts.append(f"⚠️ S_bio too high for non-biodeg ({r['s_bio']:.3f})")
                is_pass = False
        
        if is_pass:
            total_reward_pass += 1
        total_checks += 1
        
        status = "✅ PASS" if is_pass else "❌ FAIL"
        detail = "; ".join(status_parts) if status_parts else ""
        
        print(f"\n{'─' * 60}")
        print(f"  {name} [{bench['category']}] {status}")
        if detail:
            print(f"  {detail}")
        print(f"  SMILES: {bench['smiles']}")
        print(f"  S_bio:  {r['s_bio']:.4f}  (ideal: {bench['ideal_bio']:.2f}, err: {bio_err:.3f})")
        print(f"  S_mech: {r['s_mech']:.4f}  (ideal: {bench['ideal_mech']:.2f})")
        print(f"  S_syn:  {r['s_syn']:.4f}  (ideal: {bench['ideal_syn']:.2f})")
        print(f"  Reward: {r['reward']:.4f}  (min target: {bench['ideal_reward_min']:.2f})")
        if 'mech_details' in r:
            md = r['mech_details']
            print(f"  Tensile: {md['tensile']:.1f} (real: {bench['real_tensile']:.0f} MPa)")
            print(f"  Tg:      {md['tg']:.1f} (real: {bench['real_tg']:.0f} °C)")
        if r.get('polymerizability', {}).get('is_polymerizable'):
            pi = r['polymerizability']
            print(f"  Polymerizable: ✅ score={pi['score']:.2f} via {', '.join(pi['mechanisms'][:2])}")
    
    # ═══ SUMMARY ═══
    print("\n" + "═" * 80)
    print("  SUMMARY")
    print("═" * 80)
    
    bio_mae = np.mean(biodeg_errors) if biodeg_errors else 0
    non_bio_mae = np.mean(non_biodeg_errors) if non_biodeg_errors else 0
    
    print(f"\n  Passed: {total_reward_pass}/{total_checks} benchmarks")
    print(f"  S_bio MAE (biodegradable):     {bio_mae:.4f}")
    print(f"  S_bio MAE (non-biodegradable): {non_bio_mae:.4f}")
    
    # Separate biodeg vs non-biodeg average rewards
    biodeg_rewards = [results[n]['reward'] for n, b in BENCHMARKS.items() if b['category'] == 'biodegradable']
    non_biodeg_rewards = [results[n]['reward'] for n, b in BENCHMARKS.items() if b['category'] == 'non-biodegradable']
    
    print(f"\n  Avg reward (biodegradable):    {np.mean(biodeg_rewards):.4f}")
    print(f"  Avg reward (non-biodegradable): {np.mean(non_biodeg_rewards):.4f}")
    print(f"  Reward separation ratio:       {np.mean(biodeg_rewards) / max(np.mean(non_biodeg_rewards), 1e-6):.2f}x")
    
    # ═══ QUALITY ASSESSMENT ═══
    print(f"\n{'─' * 60}")
    print("  QUALITY ASSESSMENT")
    print(f"{'─' * 60}")
    
    grade = "A"
    issues = []
    
    if bio_mae > 0.15:
        grade = "C"
        issues.append("S_bio predictions need calibration (MAE > 0.15)")
    elif bio_mae > 0.08:
        grade = "B"
        issues.append("S_bio predictions are acceptable but could improve (MAE 0.08-0.15)")
    
    if np.mean(non_biodeg_rewards) > 0.30:
        grade = max(grade, "B")
        issues.append("Non-biodegradable plastics are getting too high rewards")
    
    sep = np.mean(biodeg_rewards) / max(np.mean(non_biodeg_rewards), 1e-6)
    if sep < 2.0:
        grade = max(grade, "C")
        issues.append(f"Reward separation ratio is only {sep:.1f}x (target: ≥3.0x)")
    elif sep < 3.0:
        grade = max(grade, "B")
        issues.append(f"Reward separation ratio is {sep:.1f}x (target: ≥3.0x)")
    
    if total_reward_pass / total_checks < 0.70:
        grade = "D"
        issues.append(f"Only {total_reward_pass}/{total_checks} benchmarks pass")
    
    print(f"\n  Overall Grade: {grade}")
    if issues:
        for iss in issues:
            print(f"    • {iss}")
    else:
        print("    • All metrics within ideal ranges! 🎉")
    
    # Save results as JSON
    output = {
        'grade': grade,
        'pass_rate': f"{total_reward_pass}/{total_checks}",
        'bio_mae_biodeg': round(bio_mae, 4),
        'bio_mae_nonbiodeg': round(non_bio_mae, 4),
        'avg_reward_biodeg': round(float(np.mean(biodeg_rewards)), 4),
        'avg_reward_nonbiodeg': round(float(np.mean(non_biodeg_rewards)), 4),
        'separation_ratio': round(sep, 2),
        'issues': issues,
        'details': {name: {
            's_bio': round(r['s_bio'], 4),
            's_mech': round(r['s_mech'], 4),
            's_syn': round(r['s_syn'], 4),
            'reward': round(r['reward'], 4),
        } for name, r in results.items()},
    }
    
    os.makedirs('./results', exist_ok=True)
    with open('./results/validation_report.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  📄 Full report saved to ./results/validation_report.json")
    
    return output


if __name__ == '__main__':
    device = 'cpu'  # Use CPU for validation (faster loading, no GPU needed)
    
    print("Loading trained surrogates...")
    rf = load_reward_function(device)
    
    result = validate(rf)
    
    print("\n" + "═" * 80)
    print(f"  VALIDATION COMPLETE — Grade: {result['grade']}")
    print("═" * 80 + "\n")
