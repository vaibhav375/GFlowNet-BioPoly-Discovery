"""
Molecular Dynamics Simulation Interface
==========================================
Provides a simulation layer for validating GFlowNet-generated molecules.

In a full implementation, this would interface with:
    - OpenMM / GROMACS for MD simulations
    - Gaussian / ORCA for DFT calculations
    - ASE (Atomic Simulation Environment)

For this research prototype, we implement a physics-informed surrogate
that estimates MD/DFT properties using analytical models and RDKit
descriptors, serving as a proof-of-concept for the active learning loop.

This is clearly documented as a surrogate for actual HPC simulation
and is identified as a limitation in the paper.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen

logger = logging.getLogger(__name__)


@dataclass
class MDSimulationResult:
    """Results from a molecular dynamics simulation (or surrogate)."""
    
    smiles: str
    is_stable: bool
    stability_score: float       # [0, 1]
    predicted_tg: float          # °C
    predicted_tensile: float     # MPa
    predicted_flexibility: float # [0, 1]
    predicted_biodeg_rate: float # months to 50% degradation
    energy_minimized: float      # kcal/mol
    simulation_time_ns: float    # nanoseconds simulated
    compute_time_seconds: float  # wall clock time
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'smiles': self.smiles,
            'is_stable': self.is_stable,
            'stability_score': self.stability_score,
            'predicted_tg': self.predicted_tg,
            'predicted_tensile': self.predicted_tensile,
            'predicted_flexibility': self.predicted_flexibility,
            'predicted_biodeg_rate': self.predicted_biodeg_rate,
            'energy_minimized': self.energy_minimized,
            'simulation_time_ns': self.simulation_time_ns,
            'compute_time_seconds': self.compute_time_seconds,
            'warnings': self.warnings,
        }


class MDSimulator:
    """
    Physics-informed surrogate for Molecular Dynamics simulation.
    
    This uses analytical models based on known structure-property
    relationships in polymer science to estimate properties that
    would normally require expensive MD/DFT calculations.
    
    IMPORTANT: This is a surrogate model, NOT actual MD simulation.
    The paper must clearly state this as a limitation.
    
    Physical models used:
        1. Stability: Based on strain energy, valence satisfaction
        2. Tg: Group contribution method (Van Krevelen)
        3. Tensile strength: Empirical correlation with MW, crosslinks
        4. Biodegradation: Based on hydrolyzable bond density
    """
    
    def __init__(self, noise_level: float = 0.1):
        """
        Args:
            noise_level: Gaussian noise added to predictions to simulate
                        uncertainty in real simulations
        """
        self.noise_level = noise_level
        self.simulation_count = 0
        
        # Group contribution values for Tg estimation (simplified)
        # Based on Van Krevelen's "Properties of Polymers"
        self.tg_contributions = {
            'CH2': -5.0,    # Methylene
            'CH3': -15.0,   # Methyl
            'C=O': 20.0,    # Carbonyl
            'C-O': -10.0,   # Ether
            'O-CO': 25.0,   # Ester
            'NH-CO': 40.0,  # Amide
            'phenyl': 60.0, # Aromatic ring
            'OH': 15.0,     # Hydroxyl
            'Cl': 10.0,     # Chlorine
            'F': -5.0,      # Fluorine
        }
    
    def simulate(self, smiles: str, simulation_ns: float = 10.0) -> MDSimulationResult:
        """
        Run physics-informed property prediction for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
            simulation_ns: Simulated MD time in nanoseconds
        
        Returns:
            MDSimulationResult with predicted properties
        """
        start_time = time.time()
        self.simulation_count += 1
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MDSimulationResult(
                smiles=smiles, is_stable=False, stability_score=0.0,
                predicted_tg=0.0, predicted_tensile=0.0, predicted_flexibility=0.0,
                predicted_biodeg_rate=999.0, energy_minimized=0.0,
                simulation_time_ns=0.0, compute_time_seconds=0.0,
                warnings=["Invalid SMILES"]
            )
        
        warnings = []
        
        # --- Stability Assessment ---
        stability_score, stability_warnings = self._assess_stability(mol)
        warnings.extend(stability_warnings)
        is_stable = stability_score > 0.5
        
        # --- Energy Minimization (surrogate) ---
        energy = self._estimate_energy(mol)
        
        # --- Glass Transition Temperature ---
        tg = self._predict_tg(mol)
        
        # --- Tensile Strength ---
        tensile = self._predict_tensile(mol)
        
        # --- Flexibility ---
        flexibility = self._predict_flexibility(mol)
        
        # --- Biodegradation Rate ---
        biodeg_rate = self._predict_biodeg_rate(mol)
        
        # Add noise to simulate uncertainty
        if self.noise_level > 0:
            tg += np.random.normal(0, self.noise_level * 20)
            tensile += np.random.normal(0, self.noise_level * 5)
            flexibility = np.clip(flexibility + np.random.normal(0, self.noise_level * 0.1), 0, 1)
            biodeg_rate = max(0.5, biodeg_rate + np.random.normal(0, self.noise_level * 3))
        
        compute_time = time.time() - start_time
        
        return MDSimulationResult(
            smiles=smiles,
            is_stable=is_stable,
            stability_score=stability_score,
            predicted_tg=round(tg, 1),
            predicted_tensile=round(max(0, tensile), 1),
            predicted_flexibility=round(flexibility, 3),
            predicted_biodeg_rate=round(biodeg_rate, 1),
            energy_minimized=round(energy, 2),
            simulation_time_ns=simulation_ns,
            compute_time_seconds=round(compute_time, 3),
            warnings=warnings,
        )
    
    def _assess_stability(self, mol) -> Tuple[float, List[str]]:
        """Assess molecular stability based on structural features."""
        warnings = []
        score = 1.0
        
        # Check for strained rings
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) == 3:
                score -= 0.2
                warnings.append("Contains 3-membered ring (strained)")
            elif len(ring) == 4:
                score -= 0.1
                warnings.append("Contains 4-membered ring")
        
        # Check valence violations
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            score -= 0.3
            warnings.append("Valence violation detected")
        
        # Check for reactive groups
        # Peroxide: O-O
        peroxide = Chem.MolFromSmarts('[OX2][OX2]')
        if peroxide and mol.HasSubstructMatch(peroxide):
            score -= 0.3
            warnings.append("Contains peroxide (unstable)")
        
        # Molecular weight check
        mw = Descriptors.MolWt(mol)
        if mw > 1000:
            score -= 0.1
            warnings.append(f"High molecular weight ({mw:.0f} Da)")
        
        return max(0.0, min(1.0, score)), warnings
    
    def _estimate_energy(self, mol) -> float:
        """Estimate molecular energy (kcal/mol) as a surrogate for MM optimization."""
        try:
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_3d)
            
            ff = AllChem.MMFFGetMoleculeForceField(mol_3d, AllChem.MMFFGetMoleculeProperties(mol_3d))
            if ff is not None:
                energy = ff.CalcEnergy()
                return energy
        except Exception:
            pass
        
        # Fallback: rough estimate
        return Descriptors.MolWt(mol) * 0.1
    
    def _predict_tg(self, mol) -> float:
        """
        Predict glass transition temperature using group contribution method.
        Simplified Van Krevelen approach.
        """
        tg = 20.0  # Base temperature
        
        # Count functional groups
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mw = Descriptors.MolWt(mol)
        
        # Aromatic rings increase Tg
        tg += num_aromatic * self.tg_contributions['phenyl']
        
        # Rotatable bonds decrease Tg (more flexible)
        tg -= num_rot * 5.0
        
        # Ester groups
        ester = Chem.MolFromSmarts('[#6](=[#8])-[#8]')
        if ester:
            num_esters = len(mol.GetSubstructMatches(ester))
            tg += num_esters * self.tg_contributions['O-CO']
        
        # Amide groups
        amide = Chem.MolFromSmarts('[#6](=[#8])-[#7]')
        if amide:
            num_amides = len(mol.GetSubstructMatches(amide))
            tg += num_amides * self.tg_contributions['NH-CO']
        
        # Hydroxyl groups
        hydroxyl = Chem.MolFromSmarts('[OX2H]')
        if hydroxyl:
            num_oh = len(mol.GetSubstructMatches(hydroxyl))
            tg += num_oh * self.tg_contributions['OH']
        
        # MW effect (higher MW → slightly higher Tg)
        tg += mw * 0.02
        
        return tg
    
    def _predict_tensile(self, mol) -> float:
        """Predict tensile strength (MPa) from molecular structure."""
        mw = Descriptors.MolWt(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Base tensile strength
        tensile = 10.0
        
        # Rings add strength
        tensile += num_rings * 5.0
        tensile += num_aromatic * 8.0
        
        # MW contributes to chain entanglement
        tensile += min(mw * 0.03, 20.0)
        
        # Hydrogen bonding groups add strength
        num_hbd = rdMolDescriptors.CalcNumHBD(mol)
        num_hba = rdMolDescriptors.CalcNumHBA(mol)
        tensile += min((num_hbd + num_hba) * 2.0, 15.0)
        
        return max(0, tensile)
    
    def _predict_flexibility(self, mol) -> float:
        """Predict flexibility score [0, 1]."""
        num_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_atoms = mol.GetNumHeavyAtoms()
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        if num_atoms == 0:
            return 0.5
        
        # Fraction of rotatable bonds
        rot_fraction = num_rot / max(num_atoms, 1)
        
        # Aromatic rings reduce flexibility
        flexibility = min(rot_fraction * 2.0, 0.9) - num_aromatic * 0.1
        
        return max(0.0, min(1.0, flexibility))
    
    def _predict_biodeg_rate(self, mol) -> float:
        """
        Predict biodegradation time (months to 50% degradation).
        
        Calibrated against real-world polymer literature values:
            - PGA: ~6-12 months       (very high ester density)
            - PLA: ~12-24 months       (high ester density)
            - PCL: ~24-36 months       (moderate ester density)
            - PBS: ~12-18 months       (moderate ester density)
            - PE:  ~hundreds of years  (no cleavable bonds)
            - PET: ~60-120 months      (aromatic ester)
        
        Based on:
            - Hydrolyzable bond density (esters, amides)
            - Oxygen-rich functional groups (hydroxyls, ethers)
            - Molecular weight (smaller = faster)
            - Presence of resistant groups (halogens, aromatics)
            - Chain flexibility / heteroatom content
            - Molecule size (very small molecules ≠ polymers)
        """
        num_atoms = mol.GetNumHeavyAtoms()
        if num_atoms == 0:
            return 999.0
        
        # Count hydrolyzable bonds
        ester = Chem.MolFromSmarts('[#6](=[#8])-[#8]')
        amide = Chem.MolFromSmarts('[#6](=[#8])-[#7]')
        
        num_hydrolyzable = 0
        if ester:
            num_hydrolyzable += len(mol.GetSubstructMatches(ester))
        if amide:
            num_hydrolyzable += len(mol.GetSubstructMatches(amide))
        
        # Hydrolyzable bond density
        bond_density = num_hydrolyzable / max(num_atoms, 1)
        
        # Count oxygen-containing groups that promote hydrolysis
        num_O = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        num_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N')
        num_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        o_to_c = num_O / max(num_C, 1)
        heteroatom_frac = (num_O + num_N) / max(num_atoms, 1)
        
        # Count hydroxyl groups (key for microbial accessibility)
        hydroxyl = Chem.MolFromSmarts('[OH]')
        num_oh = len(mol.GetSubstructMatches(hydroxyl)) if hydroxyl else 0
        
        # Base degradation time (months) — calibrated to literature
        if bond_density > 0.2:
            base_time = 8.0    # PGA-like (very high cleavable bond density)
        elif bond_density > 0.1:
            base_time = 14.0   # PLA-like (high ester density)
        elif bond_density > 0.05:
            base_time = 24.0   # PCL/PBS-like (moderate ester density)
        elif bond_density > 0:
            base_time = 36.0   # Sparse cleavable bonds
        elif o_to_c > 0.5 and num_oh >= 2:
            base_time = 18.0   # Oxygen-rich polyols — microbially accessible
        elif o_to_c > 0.3 or num_oh >= 1:
            base_time = 30.0   # Some oxygen content — slow-moderate biodeg
        elif heteroatom_frac > 0.3:
            base_time = 48.0   # Heteroatom-rich but no oxygen — slow
        else:
            base_time = 120.0  # Hydrocarbon-like — very slow (PE-like)
        
        # Hydroxyl bonus — each OH accelerates microbial degradation
        oh_factor = max(0.6, 1.0 - num_oh * 0.08)
        base_time *= oh_factor
        
        # Halogen penalty (persistent pollutants)
        halogens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br'])
        base_time *= (1 + halogens * 1.0)
        
        # Aromatic ring penalty — much slower hydrolysis
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        base_time *= (1 + num_aromatic * 0.6)
        
        # MW effect — larger polymers degrade slower
        mw = Descriptors.MolWt(mol)
        if mw > 500:
            base_time *= 1.0 + (mw - 500) / 800
        
        # Small molecule adjustment — monomers are NOT polymers
        # Very small molecules degrade quickly but shouldn't get
        # unrealistically fast times (< 6 months is rare even for monomers)
        if num_atoms < 8:
            base_time = max(base_time, 6.0)  # Floor at 6 months
        elif mw < 200:
            base_time *= 0.85  # Slight monomer bonus
        
        return max(3.0, base_time)
    
    def simulate_batch(
        self,
        smiles_list: List[str],
        simulation_ns: float = 10.0,
    ) -> List[MDSimulationResult]:
        """Simulate a batch of molecules."""
        results = []
        for smiles in smiles_list:
            result = self.simulate(smiles, simulation_ns)
            results.append(result)
        return results


if __name__ == "__main__":
    simulator = MDSimulator(noise_level=0.05)
    
    test_molecules = [
        ('CC(O)C(=O)O', 'Lactic acid (PLA monomer)'),
        ('CCCCCCCCCC', 'Decane (PE-like)'),
        ('COC(=O)CCCC(=O)OC', 'Dimethyl adipate (biodegradable)'),
        ('FC(F)(F)C(F)(F)F', 'Perfluoroethane (persistent)'),
        ('O=C1CCCCCO1', 'Caprolactone (PCL monomer)'),
    ]
    
    print("=" * 80)
    print("  MD Simulation Results (Physics-Informed Surrogate)")
    print("=" * 80)
    
    for smiles, name in test_molecules:
        result = simulator.simulate(smiles)
        print(f"\n{name} ({smiles})")
        print(f"  Stable: {'✅' if result.is_stable else '❌'} (score: {result.stability_score:.2f})")
        print(f"  Tg: {result.predicted_tg:.1f} °C")
        print(f"  Tensile: {result.predicted_tensile:.1f} MPa")
        print(f"  Flexibility: {result.predicted_flexibility:.3f}")
        print(f"  Biodeg time: {result.predicted_biodeg_rate:.1f} months")
        print(f"  Energy: {result.energy_minimized:.1f} kcal/mol")
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")
    
    print(f"\nTotal simulations: {simulator.simulation_count}")
    print("✓ MD Simulator ready!")
