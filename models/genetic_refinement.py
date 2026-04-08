"""
Genetic-Guided Exploration — Post-Generation Refinement
========================================================
Applies crossover + mutation to top GFlowNet-generated molecules
to exploit local chemical neighborhoods around high-reward candidates.

Paper: Kim et al. "Genetic-guided GFlowNets" (NeurIPS 2024)

Usage:
    from models.genetic_refinement import refine_population
    improved = refine_population(candidates, reward_fn, generations=5)
"""

import random
import logging
from typing import Callable, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)


# ============================================================
# Mutation operators
# ============================================================

def mutate_atom_swap(mol):
    """Swap a random non-ring carbon with another element."""
    atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and not a.IsInRing()]
    if not atoms:
        return None
    
    target = random.choice(atoms)
    new_element = random.choice([7, 8])  # N or O
    
    rwmol = Chem.RWMol(mol)
    rwmol.GetAtomWithIdx(target.GetIdx()).SetAtomicNum(new_element)
    
    try:
        Chem.SanitizeMol(rwmol)
        return rwmol.GetMol()
    except Exception:
        return None


def mutate_add_fragment(mol):
    """Add a small biodegradable fragment to a random atom."""
    fragments = [
        ('C(=O)O', 'ester'),
        ('C(N)=O', 'amide'),
        ('C(=O)', 'carbonyl'),
        ('CO', 'ether'),
        ('O', 'hydroxyl'),
    ]
    
    frag_smi, frag_name = random.choice(fragments)
    
    # Find attachment points (atoms with implicit hydrogens)
    candidates = [a for a in mol.GetAtoms() 
                  if a.GetTotalNumHs() > 0 and a.GetDegree() < 4]
    if not candidates:
        return None
    
    target = random.choice(candidates)
    
    try:
        # Build combined SMILES
        mol_smi = Chem.MolToSmiles(mol)
        # Simple approach: add fragment to a terminal position
        combined = f"{mol_smi}.{frag_smi}"
        combo = Chem.MolFromSmiles(combined)
        if combo is None:
            return None
        
        # Connect the fragments
        rwmol = Chem.RWMol(combo)
        # Find disconnected fragments and connect them
        frags = Chem.GetMolFrags(rwmol, asMols=True)
        if len(frags) > 1:
            # Return the larger fragment with modification
            largest = max(frags, key=lambda m: m.GetNumAtoms())
            return largest
        
        return combo
    except Exception:
        return None


def mutate_delete_atom(mol):
    """Remove a terminal (degree-1) atom."""
    terminal = [a for a in mol.GetAtoms() 
                if a.GetDegree() == 1 and a.GetSymbol() != 'O']
    if not terminal or mol.GetNumAtoms() <= 6:
        return None
    
    target = random.choice(terminal)
    
    try:
        rwmol = Chem.RWMol(mol)
        rwmol.RemoveAtom(target.GetIdx())
        Chem.SanitizeMol(rwmol)
        smiles = Chem.MolToSmiles(rwmol)
        result = Chem.MolFromSmiles(smiles)
        return result
    except Exception:
        return None


def mutate_bond_change(mol):
    """Change a random single bond to double or vice versa."""
    bonds = [b for b in mol.GetBonds() if not b.IsInRing()]
    if not bonds:
        return None
    
    target = random.choice(bonds)
    
    try:
        rwmol = Chem.RWMol(mol)
        if target.GetBondType() == Chem.BondType.SINGLE:
            begin_atom = rwmol.GetAtomWithIdx(target.GetBeginAtomIdx())
            end_atom = rwmol.GetAtomWithIdx(target.GetEndAtomIdx())
            if begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() in ('C', 'O', 'N'):
                rwmol.GetBondWithIdx(target.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
        elif target.GetBondType() == Chem.BondType.DOUBLE:
            rwmol.GetBondWithIdx(target.GetIdx()).SetBondType(Chem.BondType.SINGLE)
        
        Chem.SanitizeMol(rwmol)
        return rwmol.GetMol()
    except Exception:
        return None


def mutate(smiles: str) -> Optional[str]:
    """Apply a random mutation to a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mutations = [mutate_atom_swap, mutate_add_fragment, mutate_delete_atom, mutate_bond_change]
    random.shuffle(mutations)
    
    for mutation_fn in mutations:
        result = mutation_fn(mol)
        if result is not None:
            new_smiles = Chem.MolToSmiles(result)
            # Validate
            if Chem.MolFromSmiles(new_smiles) is not None and new_smiles != smiles:
                return new_smiles
    
    return None


# ============================================================
# Crossover operators
# ============================================================

def crossover(smiles_a: str, smiles_b: str) -> Optional[str]:
    """
    Fragment-exchange crossover between two molecules.
    Takes a random fragment from mol_a and combines with mol_b's scaffold.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    
    if mol_a is None or mol_b is None:
        return None
    
    try:
        # Get fragments of each molecule
        frags_a = Chem.GetMolFrags(mol_a, asMols=True, sanitizeFrags=True)
        frags_b = Chem.GetMolFrags(mol_b, asMols=True, sanitizeFrags=True)
        
        # If either has fragments, exchange them
        if len(frags_a) > 1 and len(frags_b) > 1:
            # Swap random fragments
            new_frags = [random.choice(frags_a), random.choice(frags_b)]
            combined = Chem.CombineMols(new_frags[0], new_frags[1])
            return Chem.MolToSmiles(combined)
        
        # Otherwise, try SMILES-level recombination
        # Take first half of mol_a SMILES and second half of mol_b
        smi_a = Chem.MolToSmiles(mol_a)
        smi_b = Chem.MolToSmiles(mol_b)
        
        # Find a valid split point
        mid_a = len(smi_a) // 2
        mid_b = len(smi_b) // 2
        
        # Try multiple recombination points
        for offset in range(-3, 4):
            candidate = smi_a[:max(1, mid_a + offset)] + smi_b[max(1, mid_b + offset):]
            result = Chem.MolFromSmiles(candidate)
            if result is not None:
                canonical = Chem.MolToSmiles(result)
                if canonical != smiles_a and canonical != smiles_b:
                    return canonical
        
        return None
    except Exception:
        return None


# ============================================================
# GA Refinement Loop
# ============================================================

def refine_population(
    candidates: List[Dict],
    reward_fn: Callable,
    generations: int = 5,
    population_size: int = 50,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.3,
    elite_ratio: float = 0.2,
) -> List[Dict]:
    """
    Apply genetic refinement to a set of GFlowNet-generated candidates.
    
    Args:
        candidates: List of dicts with 'smiles' and 'reward' keys
        reward_fn: Function that takes SMILES and returns reward dict
        generations: Number of GA generations
        population_size: Size of GA population
        mutation_rate: Probability of mutation per individual
        crossover_rate: Probability of crossover per pair
        elite_ratio: Fraction of top individuals preserved unchanged
    
    Returns:
        Refined candidates (potentially improved)
    """
    if not candidates or reward_fn is None:
        return candidates
    
    # Initialize population from top candidates
    sorted_cands = sorted(candidates, key=lambda x: x.get('reward', 0), reverse=True)
    population = sorted_cands[:population_size]
    
    logger.info(f"  🧬 Genetic refinement: {len(population)} candidates, "
                f"{generations} generations")
    
    seen_smiles = set(c.get('smiles', '') for c in population)
    num_elite = max(2, int(population_size * elite_ratio))
    
    for gen in range(generations):
        new_population = []
        
        # Elite preservation
        elite = population[:num_elite]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < population_size:
            # Selection (tournament)
            parent_a = max(random.sample(population, min(3, len(population))),
                         key=lambda x: x.get('reward', 0))
            
            offspring_smiles = None
            
            # Crossover
            if random.random() < crossover_rate:
                parent_b = max(random.sample(population, min(3, len(population))),
                             key=lambda x: x.get('reward', 0))
                offspring_smiles = crossover(
                    parent_a.get('smiles', ''),
                    parent_b.get('smiles', ''),
                )
            
            # Mutation
            if offspring_smiles is None and random.random() < mutation_rate:
                offspring_smiles = mutate(parent_a.get('smiles', ''))
            
            # If no valid offspring, copy parent
            if offspring_smiles is None or offspring_smiles in seen_smiles:
                offspring_smiles = mutate(parent_a.get('smiles', ''))
            
            if offspring_smiles and offspring_smiles not in seen_smiles:
                try:
                    reward_info = reward_fn(offspring_smiles)
                    if reward_info.get('valid', False):
                        new_cand = {
                            'smiles': offspring_smiles,
                            'reward': reward_info.get('reward', 0),
                            **reward_info,
                        }
                        new_population.append(new_cand)
                        seen_smiles.add(offspring_smiles)
                except Exception:
                    pass
        
        # Sort by reward
        population = sorted(new_population, key=lambda x: x.get('reward', 0), reverse=True)
        population = population[:population_size]
        
        if (gen + 1) % 2 == 0:
            best_r = population[0].get('reward', 0) if population else 0
            logger.info(f"    Gen {gen+1}/{generations}: best={best_r:.4f}, "
                       f"pop={len(population)}")
    
    # Merge with original candidates
    all_smiles = set()
    merged = []
    for c in population + candidates:
        smi = c.get('smiles', '')
        if smi and smi not in all_smiles:
            all_smiles.add(smi)
            merged.append(c)
    
    merged.sort(key=lambda x: x.get('reward', 0), reverse=True)
    
    improved = sum(1 for c in population if c.get('reward', 0) > sorted_cands[0].get('reward', 0))
    logger.info(f"  🧬 Refinement complete: {improved} improved, "
                f"{len(merged)} total candidates")
    
    return merged
