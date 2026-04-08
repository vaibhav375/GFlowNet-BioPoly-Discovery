"""
Evaluation Metrics
===================
Comprehensive metrics for evaluating GFlowNet molecule generation quality.

Metrics:
    1. Diversity (Tanimoto Distance)
    2. Validity (% chemically valid)
    3. Novelty (% not in training set)
    4. Top-K Reward (average reward of top K molecules)
    5. Internal Diversity (structural variety)
    6. Property Distribution Analysis
    7. Green AI Efficiency (molecules per kg CO2)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)


def compute_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprint for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_diversity(smiles_list: List[str], sample_size: int = 500) -> float:
    """
    Compute average pairwise Tanimoto distance (diversity metric).
    
    Diversity = 1 - average_pairwise_Tanimoto_similarity
    
    Range: [0, 1] where 1 = maximally diverse, 0 = all identical
    
    Args:
        smiles_list: List of SMILES strings
        sample_size: Max number of pairs to sample (for efficiency)
    
    Returns:
        Average Tanimoto diversity score
    """
    if len(smiles_list) < 2:
        return 0.0
    
    # Compute fingerprints
    fps = []
    for s in smiles_list:
        fp = compute_fingerprint(s)
        if fp is not None:
            fps.append(fp)
    
    if len(fps) < 2:
        return 0.0
    
    # Sample pairs for efficiency
    n = len(fps)
    if n * (n - 1) // 2 > sample_size:
        # Random sampling
        similarities = []
        for _ in range(sample_size):
            i, j = np.random.choice(n, 2, replace=False)
            sim = tanimoto_similarity(fps[i], fps[j])
            similarities.append(sim)
    else:
        # Compute all pairs
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = tanimoto_similarity(fps[i], fps[j])
                similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    diversity = 1.0 - avg_similarity
    
    return diversity


def compute_validity_rate(smiles_list: List[str]) -> float:
    """
    Compute fraction of chemically valid molecules.
    
    A molecule is valid if:
        1. RDKit can parse the SMILES
        2. Sanitization succeeds (valence check, kekulization, etc.)
    
    Returns:
        Validity rate in [0, 1]
    """
    if not smiles_list:
        return 0.0
    
    valid_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_count += 1
            except Exception:
                pass
    
    return valid_count / len(smiles_list)


def compute_novelty(
    generated_smiles: List[str],
    training_smiles: List[str],
) -> float:
    """
    Compute fraction of generated molecules NOT in training set.
    
    Novelty = |generated - training| / |generated|
    
    Returns:
        Novelty rate in [0, 1]
    """
    if not generated_smiles:
        return 0.0
    
    # Canonicalize all SMILES for fair comparison
    training_canonical = set()
    for s in training_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            training_canonical.add(Chem.MolToSmiles(mol))
    
    novel_count = 0
    for s in generated_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            if canonical not in training_canonical:
                novel_count += 1
    
    return novel_count / len(generated_smiles)


def compute_uniqueness(smiles_list: List[str]) -> float:
    """
    Compute fraction of unique molecules.
    
    Returns:
        Uniqueness rate in [0, 1]
    """
    if not smiles_list:
        return 0.0
    
    canonical = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            canonical.add(Chem.MolToSmiles(mol))
    
    return len(canonical) / len(smiles_list)


def compute_top_k_reward(rewards: List[float], k: int = 100) -> float:
    """
    Compute average reward of top-K molecules.
    
    Returns:
        Average top-K reward
    """
    if not rewards:
        return 0.0
    
    sorted_rewards = sorted(rewards, reverse=True)
    top_k = sorted_rewards[:min(k, len(sorted_rewards))]
    return np.mean(top_k)


def compute_property_statistics(smiles_list: List[str]) -> Dict:
    """
    Compute property distribution statistics for generated molecules.
    """
    properties = {
        'mol_weight': [], 'logp': [], 'num_rings': [],
        'num_aromatic_rings': [], 'num_rotatable_bonds': [],
        'tpsa': [], 'num_hba': [], 'num_hbd': [],
        'num_ester_bonds': [], 'num_atoms': [],
    }
    
    ester_pattern = Chem.MolFromSmarts('[#6](=[#8])-[#8]')
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        properties['mol_weight'].append(Descriptors.MolWt(mol))
        properties['logp'].append(Descriptors.MolLogP(mol))
        properties['num_rings'].append(rdMolDescriptors.CalcNumRings(mol))
        properties['num_aromatic_rings'].append(rdMolDescriptors.CalcNumAromaticRings(mol))
        properties['num_rotatable_bonds'].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        properties['tpsa'].append(Descriptors.TPSA(mol))
        properties['num_hba'].append(rdMolDescriptors.CalcNumHBA(mol))
        properties['num_hbd'].append(rdMolDescriptors.CalcNumHBD(mol))
        properties['num_atoms'].append(mol.GetNumHeavyAtoms())
        
        if ester_pattern is not None:
            properties['num_ester_bonds'].append(len(mol.GetSubstructMatches(ester_pattern)))
    
    # Compute stats
    stats = {}
    for key, values in properties.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
            }
    
    return stats


def compute_all_metrics(
    generated_results: List[Dict],
    training_smiles: Optional[List[str]] = None,
) -> Dict:
    """
    Compute all evaluation metrics.
    
    Args:
        generated_results: List of reward dictionaries from GFlowNet
        training_smiles: Training set SMILES for novelty computation
    
    Returns:
        Comprehensive metrics dictionary
    """
    smiles_list = [r['smiles'] for r in generated_results if r.get('smiles')]
    rewards = [r['reward'] for r in generated_results]
    
    metrics = {
        'num_generated': len(generated_results),
        'num_valid': sum(1 for r in generated_results if r.get('valid', False)),
        'validity_rate': compute_validity_rate(smiles_list),
        'uniqueness': compute_uniqueness(smiles_list),
        'diversity': compute_diversity(smiles_list),
        'mean_reward': np.mean(rewards) if rewards else 0.0,
        'std_reward': np.std(rewards) if rewards else 0.0,
        'max_reward': np.max(rewards) if rewards else 0.0,
        'top_10_reward': compute_top_k_reward(rewards, k=10),
        'top_50_reward': compute_top_k_reward(rewards, k=50),
        'top_100_reward': compute_top_k_reward(rewards, k=100),
    }
    
    # Novelty (if training set provided)
    if training_smiles is not None:
        metrics['novelty'] = compute_novelty(smiles_list, training_smiles)
    
    # Sub-scores
    bio_scores = [r.get('s_bio', 0) for r in generated_results]
    mech_scores = [r.get('s_mech', 0) for r in generated_results]
    syn_scores = [r.get('s_syn', 0) for r in generated_results]
    
    metrics['mean_s_bio'] = np.mean(bio_scores) if bio_scores else 0.0
    metrics['mean_s_mech'] = np.mean(mech_scores) if mech_scores else 0.0
    metrics['mean_s_syn'] = np.mean(syn_scores) if syn_scores else 0.0
    
    # Property statistics
    metrics['property_stats'] = compute_property_statistics(smiles_list)
    
    return metrics


def format_metrics_table(metrics: Dict) -> str:
    """Format metrics as a nice table for logging."""
    lines = []
    lines.append("=" * 50)
    lines.append("    EVALUATION METRICS")
    lines.append("=" * 50)
    
    lines.append(f"  Generated:     {metrics.get('num_generated', 0)}")
    lines.append(f"  Valid:         {metrics.get('num_valid', 0)} ({metrics.get('validity_rate', 0):.1%})")
    lines.append(f"  Uniqueness:    {metrics.get('uniqueness', 0):.1%}")
    lines.append(f"  Diversity:     {metrics.get('diversity', 0):.4f}")
    
    if 'novelty' in metrics:
        lines.append(f"  Novelty:       {metrics['novelty']:.1%}")
    
    lines.append("")
    lines.append("  --- Reward ---")
    lines.append(f"  Mean:          {metrics.get('mean_reward', 0):.4f} ± {metrics.get('std_reward', 0):.4f}")
    lines.append(f"  Max:           {metrics.get('max_reward', 0):.4f}")
    lines.append(f"  Top-10:        {metrics.get('top_10_reward', 0):.4f}")
    lines.append(f"  Top-100:       {metrics.get('top_100_reward', 0):.4f}")
    
    lines.append("")
    lines.append("  --- Sub-Scores ---")
    lines.append(f"  S_bio (avg):   {metrics.get('mean_s_bio', 0):.4f}")
    lines.append(f"  S_mech (avg):  {metrics.get('mean_s_mech', 0):.4f}")
    lines.append(f"  S_syn (avg):   {metrics.get('mean_s_syn', 0):.4f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


# ============================================================
# SESSION 9 — ENHANCED PUBLICATION METRICS
# ============================================================

def compute_qed_distribution(smiles_list: List[str]) -> Dict[str, float]:
    """
    Compute Quantitative Estimate of Drug-likeness (QED) distribution.
    Higher QED = more drug-like / practical molecule.
    Range: 0-1, good drug candidates > 0.5
    """
    from rdkit.Chem import QED as QED_module
    qed_scores = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                qed = QED_module.qed(mol)
                qed_scores.append(qed)
            except Exception:
                pass
    if not qed_scores:
        return {'mean_qed': 0.0, 'std_qed': 0.0, 'median_qed': 0.0, 'pct_above_0.5': 0.0}
    return {
        'mean_qed': float(np.mean(qed_scores)),
        'std_qed': float(np.std(qed_scores)),
        'median_qed': float(np.median(qed_scores)),
        'pct_above_0.5': float(np.mean([1  if q > 0.5 else 0 for q in qed_scores])),
    }


def compute_scaffold_diversity(smiles_list: List[str]) -> Dict[str, float]:
    """
    Compute Murcko scaffold diversity.
    Scaffold = generic molecular framework after removing side chains.
    Higher unique scaffold ratio = more structurally diverse.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = set()
    valid = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                scaffolds.add(scaffold)
                valid += 1
            except Exception:
                pass
    if valid == 0:
        return {'unique_scaffolds': 0, 'scaffold_ratio': 0.0}
    return {
        'unique_scaffolds': len(scaffolds),
        'scaffold_ratio': float(len(scaffolds) / valid),
    }


def compute_nn_novelty(
    generated_smiles: List[str],
    reference_smiles: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Dict[str, float]:
    """
    Compute nearest-neighbor novelty.
    For each generated molecule, find closest match in reference set.
    High novelty = far from all reference molecules.
    """
    gen_fps = []
    for s in generated_smiles:
        fp = compute_fingerprint(s, radius, n_bits)
        if fp is not None:
            gen_fps.append(fp)
    
    ref_fps = []
    for s in reference_smiles:
        fp = compute_fingerprint(s, radius, n_bits)
        if fp is not None:
            ref_fps.append(fp)
    
    if not gen_fps or not ref_fps:
        return {'mean_nn_distance': 0.0, 'pct_novel_0.7': 0.0}
    
    nn_distances = []
    for gfp in gen_fps:
        max_sim = max(DataStructs.TanimotoSimilarity(gfp, rfp) for rfp in ref_fps)
        nn_distances.append(1.0 - max_sim)  # Distance = 1 - similarity
    
    return {
        'mean_nn_distance': float(np.mean(nn_distances)),
        'min_nn_distance': float(np.min(nn_distances)),
        'max_nn_distance': float(np.max(nn_distances)),
        'pct_novel_0.7': float(np.mean([1 if d > 0.3 else 0 for d in nn_distances])),
    }


# ── SESSION 11: D1 — SA Score Distribution ───────────────────────

def compute_sa_distribution(smiles_list: list) -> dict:
    """
    Compute Synthetic Accessibility score distribution.
    
    SA scores range from 1 (easy to synthesize) to 10 (hard).
    Good polymer candidates should have SA < 5.
    
    Returns:
        Dict with mean, median, std, and percentage below thresholds.
    """
    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    try:
        import sascorer
    except ImportError:
        # Fallback: estimate SA from molecular features
        sa_scores = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            # Simplified SA estimation
            n_atoms = mol.GetNumHeavyAtoms()
            n_rings = mol.GetRingInfo().NumRings()
            n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            sa_est = 1.0 + 0.2 * n_atoms + 0.5 * n_stereo - 0.3 * min(n_rings, 3)
            sa_scores.append(max(1.0, min(10.0, sa_est)))
        
        if not sa_scores:
            return {'mean_sa': 0.0, 'median_sa': 0.0, 'std_sa': 0.0, 'pct_below_4': 0.0, 'pct_below_6': 0.0}
        
        return {
            'mean_sa': float(np.mean(sa_scores)),
            'median_sa': float(np.median(sa_scores)),
            'std_sa': float(np.std(sa_scores)),
            'pct_below_4': float(np.mean([1 if s < 4 else 0 for s in sa_scores])),
            'pct_below_6': float(np.mean([1 if s < 6 else 0 for s in sa_scores])),
        }
    
    sa_scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            sa = sascorer.calculateScore(mol)
            sa_scores.append(sa)
        except Exception:
            continue
    
    if not sa_scores:
        return {'mean_sa': 0.0, 'median_sa': 0.0, 'std_sa': 0.0, 'pct_below_4': 0.0, 'pct_below_6': 0.0}
    
    return {
        'mean_sa': float(np.mean(sa_scores)),
        'median_sa': float(np.median(sa_scores)),
        'std_sa': float(np.std(sa_scores)),
        'pct_below_4': float(np.mean([1 if s < 4 else 0 for s in sa_scores])),
        'pct_below_6': float(np.mean([1 if s < 6 else 0 for s in sa_scores])),
    }


# ── SESSION 11: D2 — Property Correlation Analysis ──────────────

def compute_property_correlations(s_bio_scores: list, s_mech_scores: list, s_syn_scores: list) -> dict:
    """
    Compute Spearman rank correlations between property scores.
    
    Good multi-objective optimization should show weak or negative
    correlations between objectives (trade-offs, not redundancy).
    
    Returns:
        Dict with pairwise Spearman correlations and p-values.
    """
    from scipy import stats
    
    results = {}
    
    if len(s_bio_scores) >= 3 and len(s_mech_scores) >= 3:
        min_len = min(len(s_bio_scores), len(s_mech_scores))
        rho, p = stats.spearmanr(s_bio_scores[:min_len], s_mech_scores[:min_len])
        results['bio_mech_rho'] = float(rho)
        results['bio_mech_p'] = float(p)
    
    if len(s_bio_scores) >= 3 and len(s_syn_scores) >= 3:
        min_len = min(len(s_bio_scores), len(s_syn_scores))
        rho, p = stats.spearmanr(s_bio_scores[:min_len], s_syn_scores[:min_len])
        results['bio_syn_rho'] = float(rho)
        results['bio_syn_p'] = float(p)
    
    if len(s_mech_scores) >= 3 and len(s_syn_scores) >= 3:
        min_len = min(len(s_mech_scores), len(s_syn_scores))
        rho, p = stats.spearmanr(s_mech_scores[:min_len], s_syn_scores[:min_len])
        results['mech_syn_rho'] = float(rho)
        results['mech_syn_p'] = float(p)
    
    # Interpretation: |rho| < 0.3 = weak, good for multi-objective
    strong_corr = sum(1 for k, v in results.items() if k.endswith('_rho') and abs(v) > 0.5)
    results['n_strongly_correlated_pairs'] = strong_corr
    results['multi_obj_quality'] = 'good' if strong_corr == 0 else ('moderate' if strong_corr == 1 else 'poor')
    
    return results


if __name__ == "__main__":
    # Quick test with some molecules
    test_smiles = [
        'CC(O)C(=O)O',
        'OCC(=O)O',
        'CC(=O)OCC',
        'CCCCCCCCCC',
        'c1ccccc1',
        'COC(=O)CCCC(=O)OC',
        'CC(OC(=O)C(C)O)C(=O)O',
        'O=C1CCCCCO1',
        'OCCCCO',
        'CC(O)CC(=O)O',
    ]
    
    print("Testing metrics with sample molecules...")
    print(f"  Validity: {compute_validity_rate(test_smiles):.1%}")
    print(f"  Uniqueness: {compute_uniqueness(test_smiles):.1%}")
    print(f"  Diversity: {compute_diversity(test_smiles):.4f}")
    
    stats = compute_property_statistics(test_smiles)
    print(f"\nProperty Statistics:")
    for key, val in stats.items():
        print(f"  {key}: mean={val['mean']:.2f}, std={val['std']:.2f}")
    
    print("\n✓ Metrics module ready!")
