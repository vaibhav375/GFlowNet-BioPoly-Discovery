"""
Surrogate Model: S_syn — Synthesizability Scorer
==================================================
Computes the Synthetic Accessibility Score (SA Score) for molecules.

Based on: Ertl & Schuffenhauer (2009)
"Estimation of synthetic accessibility score of drug-like molecules
based on molecular complexity and fragment contributions"

SA Score ranges from 1 (easy to synthesize) to 10 (very difficult).
We normalize to [0, 1] where 1 = easy, 0 = impossible.
"""

import math
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem


def calculate_sa_score(mol) -> float:
    """
    Calculate the Synthetic Accessibility Score for an RDKit molecule.
    
    This is a simplified implementation based on:
    - Fragment contributions (presence of common/uncommon fragments)
    - Molecular complexity (ring systems, stereocenters, MW)
    - Symmetry
    
    Returns:
        SA Score in [1, 10] where 1 = easy, 10 = very hard
    """
    if mol is None:
        return 10.0
    
    # --- Component 1: Fragment Score ---
    # Common fragments in drug-like / synthesizable molecules
    # Higher score for uncommon fragments
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = gen.GetCountFingerprint(mol)
        fps = fp.GetNonzeroElements()
    except Exception:
        fp = AllChem.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
    
    # Heuristic: molecules with more unique fingerprint features are harder
    num_features = len(fps)
    fragment_score = 1.0 + min(num_features / 50.0, 3.0)  # [1, 4]
    
    # --- Component 2: Complexity Score ---
    num_atoms = mol.GetNumHeavyAtoms()
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    
    complexity_score = (
        0.5 +
        num_rings * 0.3 +
        num_stereo * 0.5 +
        num_spiro * 1.0 +
        num_bridgehead * 1.0 +
        (num_atoms / 40.0)
    )
    complexity_score = min(complexity_score, 4.0)  # Cap at 4
    
    # --- Component 3: Ring Complexity ---
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    
    ring_penalty = 0.0
    for size in ring_sizes:
        if size < 5 or size > 7:
            ring_penalty += 0.5  # Unusual ring sizes are harder
        if size > 8:
            ring_penalty += 1.0  # Large rings are very hard
    
    ring_penalty = min(ring_penalty, 2.0)
    
    # --- Component 4: Symmetry Bonus ---
    # Symmetric molecules are easier to make
    try:
        smiles = Chem.MolToSmiles(mol)
        # Simple symmetry heuristic
        n = len(smiles)
        half = smiles[:n//2]
        symmetry_bonus = 0.0
        if n > 4:
            # Check for repeated patterns
            for pat_len in range(2, min(n//2, 10)):
                pattern = smiles[:pat_len]
                count = smiles.count(pattern)
                if count > 2:
                    symmetry_bonus = min(count * 0.1, 0.5)
                    break
    except Exception:
        symmetry_bonus = 0.0
    
    # Combine components
    sa_score = fragment_score + complexity_score + ring_penalty - symmetry_bonus
    
    # Clamp to [1, 10]
    sa_score = max(1.0, min(10.0, sa_score))
    
    return sa_score


def normalize_sa_score(sa_score: float) -> float:
    """
    Normalize SA Score from [1, 10] to [0, 1] where 1 = easy to synthesize.
    
    Uses linear mapping with a slight bias toward easy molecules:
        S_syn = max(0, 1 - (SA - 1) / 9)  then softly clamped.
    
    This gives chemically meaningful scores:
        SA=1   → S_syn ≈ 1.00  (trivially easy — e.g. ethanol)
        SA=2   → S_syn ≈ 0.89  (very easy — lactic acid, glycolic acid)
        SA=3   → S_syn ≈ 0.78  (easy — most biodegradable monomers)
        SA=3.5 → S_syn ≈ 0.72  (moderately easy — PLA dimer)
        SA=5   → S_syn ≈ 0.56  (moderate — acceptable threshold)
        SA=7   → S_syn ≈ 0.33  (hard — some specialty monomers)
        SA=10  → S_syn ≈ 0.00  (nearly impossible)
    
    Per user spec: SA_score 1.5-3.5 is ideal → S_syn 0.72-0.94
    Minimum threshold S_syn ≥ 0.5 means SA_score < 5.5
    """
    # Linear mapping: 1→1.0, 10→0.0
    score = max(0.0, 1.0 - (sa_score - 1.0) / 9.0)
    return score


def compute_synthesizability(smiles: str) -> float:
    """
    Compute normalized synthesizability score for a SMILES string.
    
    Args:
        smiles: SMILES string
    
    Returns:
        Synthesizability score in [0, 1] where 1 = easy
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    
    sa_raw = calculate_sa_score(mol)
    return normalize_sa_score(sa_raw)


def compute_synthesizability_batch(smiles_list: list) -> list:
    """Compute synthesizability for a batch of SMILES."""
    return [compute_synthesizability(s) for s in smiles_list]


class SynthesizabilityScorer:
    """
    Wrapper class for synthesizability scoring with caching.
    Avoids recomputing scores for molecules seen before.
    """
    
    def __init__(self):
        self._cache = {}
    
    def score(self, smiles: str) -> float:
        """Get synthesizability score with caching."""
        if smiles not in self._cache:
            self._cache[smiles] = compute_synthesizability(smiles)
        return self._cache[smiles]
    
    def score_batch(self, smiles_list: list) -> list:
        """Score a batch of molecules."""
        return [self.score(s) for s in smiles_list]
    
    def get_sa_score(self, smiles: str) -> float:
        """Get raw SA Score (1-10 scale)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        return calculate_sa_score(mol)
    
    def clear_cache(self):
        """Clear the score cache."""
        self._cache.clear()
    
    @property
    def cache_size(self):
        return len(self._cache)


if __name__ == "__main__":
    scorer = SynthesizabilityScorer()
    
    test_molecules = [
        ('CC(O)C(=O)O', 'Lactic acid (PLA monomer)'),
        ('OCC(=O)O', 'Glycolic acid (PGA monomer)'),
        ('O=C1CCCCCO1', 'Caprolactone (PCL monomer)'),
        ('CCCCCCCCCC', 'Decane (PE-like)'),
        ('c1ccc(C(c2ccccc2)(C)C)cc1', 'Bisphenol A core'),
        ('FC(F)(F)C(F)(F)F', 'Perfluoroethane'),
        ('OC(=O)CCC(=O)O', 'Succinic acid'),
        ('CC(OC(=O)C(C)O)C(=O)O', 'Lactide dimer'),
    ]
    
    print("=" * 70)
    print("Synthesizability Scores (S_syn)")
    print("=" * 70)
    print(f"{'Molecule':<35} {'SA Raw':>8} {'S_syn':>8}")
    print("-" * 70)
    
    for smiles, name in test_molecules:
        sa_raw = scorer.get_sa_score(smiles)
        s_syn = scorer.score(smiles)
        print(f"{name:<35} {sa_raw:>8.2f} {s_syn:>8.4f}")
    
    print(f"\nCache size: {scorer.cache_size}")
    print("✓ S_syn scorer ready!")
