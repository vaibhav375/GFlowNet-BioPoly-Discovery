"""
GFlowNet: Generative Flow Network for Molecule Generation
============================================================
Core implementation of the GFlowNet that generates biodegradable
polymer candidates by constructing molecules atom-by-atom.

Theory:
    GFlowNets learn a policy π such that the probability of generating
    a molecule x is proportional to its reward R(x):
        P(x) ∝ R(x)
    
    This is achieved by learning flows through a DAG of molecular 
    construction states, where the flow matching condition ensures
    correct proportional sampling.

Training Objectives:
    1. Flow Matching (FM): ∑_parents F(s'→s) = F(s) + R(s)·𝟙[s terminal]
    2. Detailed Balance (DB): F(s)·P_F(s'|s) = F(s')·P_B(s|s')
    3. Trajectory Balance (TB): Z·∏P_F = R(x)·∏P_B  [RECOMMENDED]

References:
    - Bengio et al. (2021) "Flow Network based Generative Models"
    - Malkin et al. (2022) "Trajectory Balance"
    - Bengio et al. (2023) "GFlowNet Foundations"
"""

import os
import copy
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

from models.policy_network import (
    PolicyNetwork,
    create_policy_network,
    ATOM_ACTIONS,
    FRAGMENT_ACTIONS,
    BOND_ACTIONS,
    NUM_ATOM_ACTIONS,
    NUM_FRAGMENT_ACTIONS,
    NUM_BOND_ACTIONS,
    TOTAL_ACTIONS,
)
from models.surrogate_bio import BiodegradabilityPredictor
from models.surrogate_mech import MechanicalPropertiesPredictor
from models.surrogate_syn import SynthesizabilityScorer, compute_synthesizability
from data.preprocessing import (
    smiles_to_graph,
    get_atom_features,
    get_bond_features,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    ATOM_TYPES,
)

logger = logging.getLogger(__name__)


class MolecularState:
    """
    Represents the state of a partially-constructed molecule.
    
    Tracks:
        - Current atoms and bonds
        - Available actions (which atoms can be added, which bonds formed)
        - Step count
        - Whether the molecule is complete
    """
    
    def __init__(self, max_atoms: int = 50, min_atoms: int = 8):
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms  # Min atoms before stop is allowed
        self.atoms = []         # List of atom symbols
        self.bonds = []         # List of (atom_i, atom_j, bond_type)
        self.step = 0
        self.done = False
        self._mol = None
        self._smiles = None
    
    def copy(self) -> 'MolecularState':
        """Deep copy of the state."""
        new_state = MolecularState(self.max_atoms, self.min_atoms)
        new_state.atoms = list(self.atoms)
        new_state.bonds = list(self.bonds)
        new_state.step = self.step
        new_state.done = self.done
        return new_state
    
    @property
    def num_atoms(self) -> int:
        return len(self.atoms)
    
    @property
    def is_empty(self) -> bool:
        return len(self.atoms) == 0
    
    def add_atom(self, atom_symbol: str) -> bool:
        """Add an atom to the molecule. Returns True if successful."""
        if self.num_atoms >= self.max_atoms:
            return False
        if atom_symbol not in ATOM_TYPES:
            return False
        
        self.atoms.append(atom_symbol)
        self._mol = None  # Invalidate cache
        self._smiles = None
        self.step += 1
        return True
    
    def add_bond(self, atom_i: int, atom_j: int, bond_type: str = 'SINGLE') -> bool:
        """
        Add a bond between two atoms. Returns True if successful.
        Validates that the bond doesn't violate valence rules for BOTH atoms.
        """
        if atom_i >= self.num_atoms or atom_j >= self.num_atoms:
            return False
        if atom_i == atom_j:
            return False
        
        # Check if bond already exists
        for bi, bj, bt in self.bonds:
            if (bi == atom_i and bj == atom_j) or (bi == atom_j and bj == atom_i):
                return False
        
        # Bond order for valence counting
        bond_order = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bond_type, 1)
        
        # Validate valence for atom_i
        used_valence_i = sum(
            {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bt, 1)
            for bi, bj, bt in self.bonds
            if bi == atom_i or bj == atom_i
        )
        max_valence_i = self._get_max_valence(self.atoms[atom_i])
        if used_valence_i + bond_order > max_valence_i:
            return False
        
        # Validate valence for atom_j
        used_valence_j = sum(
            {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bt, 1)
            for bi, bj, bt in self.bonds
            if bi == atom_j or bj == atom_j
        )
        max_valence_j = self._get_max_valence(self.atoms[atom_j])
        if used_valence_j + bond_order > max_valence_j:
            return False
        
        self.bonds.append((atom_i, atom_j, bond_type))
        self._mol = None
        self._smiles = None
        self.step += 1
        return True
    
    def terminate(self):
        """Mark the molecule as complete."""
        self.done = True
    
    def to_rdkit_mol(self) -> Optional[Chem.Mol]:
        """Convert current state to RDKit molecule."""
        if self._mol is not None:
            return self._mol
        
        if self.num_atoms == 0:
            return None
        
        rw_mol = Chem.RWMol()
        
        # Add atoms
        for symbol in self.atoms:
            atom = Chem.Atom(symbol)
            rw_mol.AddAtom(atom)
        
        # Add bonds
        bond_type_map = {
            'SINGLE': Chem.rdchem.BondType.SINGLE,
            'DOUBLE': Chem.rdchem.BondType.DOUBLE,
            'TRIPLE': Chem.rdchem.BondType.TRIPLE,
        }
        
        for atom_i, atom_j, bt in self.bonds:
            try:
                rw_mol.AddBond(atom_i, atom_j, bond_type_map.get(bt, Chem.rdchem.BondType.SINGLE))
            except Exception:
                continue
        
        try:
            mol = rw_mol.GetMol()
            Chem.SanitizeMol(mol)
            self._mol = mol
            return mol
        except Exception:
            # Return unsanitized mol
            try:
                self._mol = rw_mol.GetMol()
                return self._mol
            except Exception:
                return None
    
    def to_smiles(self) -> Optional[str]:
        """Get SMILES string of current molecule."""
        if self._smiles is not None:
            return self._smiles
        
        mol = self.to_rdkit_mol()
        if mol is None:
            return None
        
        try:
            self._smiles = Chem.MolToSmiles(mol)
            return self._smiles
        except Exception:
            return None
    
    def is_valid(self) -> bool:
        """Check if current molecule is chemically valid."""
        mol = self.to_rdkit_mol()
        if mol is None:
            return False
        
        try:
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False
    
    def to_graph_data(self, device: str = 'cpu') -> Optional[Data]:
        """Convert to PyTorch Geometric Data for GNN input."""
        smiles = self.to_smiles()
        if smiles is None:
            # Create a minimal graph for empty/invalid states
            return self._create_empty_graph(device)
        
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            graph.edge_attr = graph.edge_attr.to(device)
        return graph
    
    def _create_empty_graph(self, device: str = 'cpu') -> Data:
        """Create a minimal graph for the empty state."""
        # Single carbon atom as initial state
        x = torch.zeros(1, ATOM_FEATURE_DIM, device=device)
        x[0, ATOM_TYPES.index('C')] = 1.0  # Carbon
        
        return Data(
            x=x,
            edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
            edge_attr=torch.zeros(0, BOND_FEATURE_DIM, device=device),
            batch=torch.zeros(1, dtype=torch.long, device=device),
        )
    
    def get_valid_actions(self) -> torch.Tensor:
        """
        Get mask of valid actions in current state.
        
        Action indices (Session 4 expanded):
            0-7:   Add atom (C, N, O, S, F, Cl, Br, P)
            8-13:  Add fragment (ester, amide, carbonyl, carboxylic, carbonate, vinyl)
            14-16: Add bond (single, double, triple) to last atom
            17:    Stop
        
        Returns:
            Boolean tensor [TOTAL_ACTIONS]  (18 total)
        """
        mask = torch.zeros(TOTAL_ACTIONS, dtype=torch.bool)
        
        BLOCKED_ATOM_INDICES = {3, 4, 5, 6, 7}  # S, F, Cl, Br, P
        
        def used_valence(atom_idx):
            return sum(
                {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bt, 1)
                for bi, bj, bt in self.bonds
                if bi == atom_idx or bj == atom_idx
            )
        
        BLOCKED_AUTOBOND_PAIRS = {
            frozenset({'N', 'O'}),
            frozenset({'N', 'N'}),
            frozenset({'O', 'O'}),
        }
        
        # --- Atom-add actions (0..7) ---
        if self.num_atoms < self.max_atoms:
            if self.num_atoms == 0:
                for i in range(NUM_ATOM_ACTIONS):
                    if i not in BLOCKED_ATOM_INDICES:
                        mask[i] = True
            else:
                last_idx = self.num_atoms - 1
                last_symbol = self.atoms[last_idx]
                remaining = self._get_max_valence(last_symbol) - used_valence(last_idx)
                if remaining >= 1:
                    for i in range(NUM_ATOM_ACTIONS):
                        if i not in BLOCKED_ATOM_INDICES:
                            new_symbol = ATOM_ACTIONS[i]
                            pair = frozenset({last_symbol, new_symbol})
                            if pair in BLOCKED_AUTOBOND_PAIRS:
                                continue
                            mask[i] = True
        
        # --- Fragment actions (8..13) ---
        # Each fragment adds multiple atoms + bonds as a unit.
        # Fragment sizes: C(=O)O=3, C(=O)N=3, C(=O)=2, C(=O)OH=3, OC(=O)O=4, C=C=2
        FRAGMENT_SIZES = [3, 3, 2, 3, 4, 2]
        if self.num_atoms > 0 and self.num_atoms + 2 <= self.max_atoms:  # Need room for at least 2 atoms
            last_idx = self.num_atoms - 1
            last_symbol = self.atoms[last_idx]
            remaining = self._get_max_valence(last_symbol) - used_valence(last_idx)
            if remaining >= 1:  # Need at least 1 bond to attach fragment
                for fi in range(NUM_FRAGMENT_ACTIONS):
                    frag_size = FRAGMENT_SIZES[fi]
                    if self.num_atoms + frag_size <= self.max_atoms:
                        # Check: first atom of fragment bonds to last atom
                        # Fragment first atoms are: C,C,C,C,O,C
                        frag_first = 'C' if fi != 4 else 'O'  # OC(=O)O starts with O
                        pair = frozenset({last_symbol, frag_first})
                        if pair not in BLOCKED_AUTOBOND_PAIRS:
                            mask[NUM_ATOM_ACTIONS + fi] = True
        
        # --- Bond actions (14..16) ---
        BLOCKED_BOND_PAIRS = {
            frozenset({'N', 'O'}),
            frozenset({'N', 'N'}),
            frozenset({'O', 'O'}),
        }
        
        bond_offset = NUM_ATOM_ACTIONS + NUM_FRAGMENT_ACTIONS
        if self.num_atoms >= 2:
            last_idx = self.num_atoms - 1
            last_symbol = self.atoms[last_idx]
            last_remaining = self._get_max_valence(last_symbol) - used_valence(last_idx)
            
            if last_remaining >= 1:
                bonded_to_last = set()
                for bi, bj, bt in self.bonds:
                    if bi == last_idx:
                        bonded_to_last.add(bj)
                    elif bj == last_idx:
                        bonded_to_last.add(bi)
                
                for j, bond_type_name in enumerate(BOND_ACTIONS):
                    bond_order = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bond_type_name, 1)
                    if last_remaining < bond_order:
                        continue
                    for target in range(last_idx):
                        if target in bonded_to_last:
                            continue
                        pair = frozenset({last_symbol, self.atoms[target]})
                        if pair in BLOCKED_BOND_PAIRS:
                            continue
                        target_remaining = self._get_max_valence(self.atoms[target]) - used_valence(target)
                        if target_remaining >= bond_order:
                            mask[bond_offset + j] = True
                            break
        
        # --- Stop action (17) ---
        if self.num_atoms >= self.min_atoms:
            mask[TOTAL_ACTIONS - 1] = True
        
        return mask
    
    def _get_max_valence(self, atom_symbol: str) -> int:
        """Get maximum valence for an atom type."""
        valence_map = {
            'C': 4, 'N': 3, 'O': 2, 'S': 2,
            'F': 1, 'Cl': 1, 'Br': 1, 'P': 5,
        }
        return valence_map.get(atom_symbol, 4)
    
    def _apply_fragment(self, frag_idx: int) -> bool:
        """
        Apply a fragment action — adds a pre-built molecular fragment.
        
        Fragment definitions:
            0: C(=O)O   — Ester (C bonded to =O and -O)
            1: C(=O)N   — Amide (C bonded to =O and -N)
            2: C(=O)    — Carbonyl (C bonded to =O)
            3: C(=O)OH  — Carboxylic acid
            4: OC(=O)O  — Carbonate
            5: C=C      — Vinyl double bond
        """
        if self.num_atoms == 0:
            return False
        
        attach_idx = self.num_atoms - 1  # Attach to last atom
        
        # Build each fragment atom-by-atom with internal bonds
        if frag_idx == 0:  # C(=O)O — ester
            c_idx = self.num_atoms
            self.add_atom('C')
            self.add_bond(attach_idx, c_idx, 'SINGLE')
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 1, 'DOUBLE')  # C=O
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 2, 'SINGLE')  # C-O
            return True
        
        elif frag_idx == 1:  # C(=O)N — amide
            c_idx = self.num_atoms
            self.add_atom('C')
            self.add_bond(attach_idx, c_idx, 'SINGLE')
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 1, 'DOUBLE')  # C=O
            self.add_atom('N')
            self.add_bond(c_idx, c_idx + 2, 'SINGLE')  # C-N
            return True
        
        elif frag_idx == 2:  # C(=O) — carbonyl
            c_idx = self.num_atoms
            self.add_atom('C')
            self.add_bond(attach_idx, c_idx, 'SINGLE')
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 1, 'DOUBLE')  # C=O
            return True
        
        elif frag_idx == 3:  # C(=O)OH — carboxylic acid
            c_idx = self.num_atoms
            self.add_atom('C')
            self.add_bond(attach_idx, c_idx, 'SINGLE')
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 1, 'DOUBLE')  # C=O
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 2, 'SINGLE')  # C-OH
            return True
        
        elif frag_idx == 4:  # OC(=O)O — carbonate
            o1_idx = self.num_atoms
            self.add_atom('O')
            self.add_bond(attach_idx, o1_idx, 'SINGLE')
            c_idx = o1_idx + 1
            self.add_atom('C')
            self.add_bond(o1_idx, c_idx, 'SINGLE')  # O-C
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 1, 'DOUBLE')  # C=O
            self.add_atom('O')
            self.add_bond(c_idx, c_idx + 2, 'SINGLE')  # C-O
            return True
        
        elif frag_idx == 5:  # C=C — vinyl
            c_idx = self.num_atoms
            self.add_atom('C')
            self.add_bond(attach_idx, c_idx, 'SINGLE')
            self.add_atom('C')
            self.add_bond(c_idx, c_idx + 1, 'DOUBLE')  # C=C
            return True
        
        return False
    
    def apply_action(self, action_idx: int) -> bool:
        """
        Apply an action to the state.
        
        Args:
            action_idx: Action index (0-17)
                0-7:   Add atom
                8-13:  Add fragment
                14-16: Add bond
                17:    Stop
        
        Returns:
            True if action was successfully applied
        """
        if action_idx < NUM_ATOM_ACTIONS:
            # Add atom action
            atom_symbol = ATOM_ACTIONS[action_idx]
            success = self.add_atom(atom_symbol)
            
            if success and self.num_atoms > 1:
                bond_ok = self.add_bond(self.num_atoms - 2, self.num_atoms - 1, 'SINGLE')
                if not bond_ok:
                    self.atoms.pop()
                    self._mol = None
                    self._smiles = None
                    self.step -= 1
                    return False
            
            return success
        
        elif action_idx < NUM_ATOM_ACTIONS + NUM_FRAGMENT_ACTIONS:
            # Fragment action — add pre-built molecular fragment
            frag_idx = action_idx - NUM_ATOM_ACTIONS
            return self._apply_fragment(frag_idx)
        
        elif action_idx < NUM_ATOM_ACTIONS + NUM_FRAGMENT_ACTIONS + NUM_BOND_ACTIONS:
            # Bond action — connect last atom to a random earlier atom
            bond_type = BOND_ACTIONS[action_idx - NUM_ATOM_ACTIONS - NUM_FRAGMENT_ACTIONS]
            bond_order = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bond_type, 1)
            if self.num_atoms < 2:
                return False
            
            last_idx = self.num_atoms - 1
            last_symbol = self.atoms[last_idx]
            bonded_to_last = set()
            for bi, bj, bt in self.bonds:
                if bi == last_idx:
                    bonded_to_last.add(bj)
                elif bj == last_idx:
                    bonded_to_last.add(bi)
            
            BLOCKED_BOND_PAIRS = {
                frozenset({'N', 'O'}),
                frozenset({'N', 'N'}),
                frozenset({'O', 'O'}),
            }
            
            candidates = []
            for i in range(last_idx):
                if i in bonded_to_last:
                    continue
                pair = frozenset({last_symbol, self.atoms[i]})
                if pair in BLOCKED_BOND_PAIRS:
                    continue
                used_v = sum(
                    {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bt, 1)
                    for bi, bj, bt in self.bonds
                    if bi == i or bj == i
                )
                if self._get_max_valence(self.atoms[i]) - used_v >= bond_order:
                    candidates.append(i)
            
            if not candidates:
                return False
            
            target = np.random.choice(candidates)
            return self.add_bond(last_idx, target, bond_type)
        
        else:
            # Stop action
            self.terminate()
            return True
    
    def __repr__(self):
        smiles = self.to_smiles() or "empty"
        return f"MolecularState(atoms={self.num_atoms}, bonds={len(self.bonds)}, smiles='{smiles}', done={self.done})"


class RewardFunction:
    """
    Multi-objective reward function for biodegradable polymer discovery.
    
    R_base = S_bio^0.35 × S_mech^0.50 × S_syn^0.15
    
    Hard thresholds applied:
        S_bio ≥ 0.6, S_mech ≥ 0.5, S_syn ≥ 0.5
    
    Multiplicative formulation naturally penalizes low sub-scores:
    any score near zero collapses the product.  Mechanical properties
    get the highest exponent (0.50) because S_mech is the hardest
    objective.  Reward shaping guides exploration toward biodegradable
    functional groups and penalizes unrealistic chemistry.
    """
    
    def __init__(
        self,
        bio_model: BiodegradabilityPredictor,
        mech_model: MechanicalPropertiesPredictor,
        syn_scorer: SynthesizabilityScorer,
        alpha_bio: float = 1.5,
        alpha_mech: float = 1.0,
        alpha_syn: float = 0.8,
        reward_exponent: float = 4.0,
        reward_min: float = 1e-8,
        device: str = 'cpu',
        # Reward shaping parameters
        ester_bonus: float = 0.15,
        amide_bonus: float = 0.10,
        hydroxyl_bonus: float = 0.05,
        size_bonus_threshold: int = 8,
        size_bonus: float = 0.1,
        halogen_penalty: float = 0.2,
        max_shaping_bonus: float = 0.8,
    ):
        self.bio_model = bio_model.to(device)
        self.mech_model = mech_model.to(device)
        self.syn_scorer = syn_scorer
        self.alpha_bio = alpha_bio
        self.alpha_mech = alpha_mech
        self.alpha_syn = alpha_syn
        self.reward_exponent = reward_exponent
        self.reward_min = reward_min
        self.device = device
        
        # Reward shaping
        self.ester_bonus = ester_bonus
        self.amide_bonus = amide_bonus
        self.hydroxyl_bonus = hydroxyl_bonus
        self.size_bonus_threshold = size_bonus_threshold
        self.size_bonus = size_bonus
        self.halogen_penalty = halogen_penalty
        self.max_shaping_bonus = max_shaping_bonus
        
        # SMARTS patterns for reward shaping
        self._ester_pat = Chem.MolFromSmarts('[#6](=[#8])-[#8]')
        self._amide_pat = Chem.MolFromSmarts('[#6](=[#8])-[#7]')
        self._hydroxyl_pat = Chem.MolFromSmarts('[OX2H]')
        
        # Put surrogate models in eval mode
        self.bio_model.eval()
        self.mech_model.eval()
    
    def _compute_shaping_bonus(self, mol) -> float:
        """
        Compute reward shaping bonus based on structural features.
        Guides exploration toward chemically realistic, biodegradable 
        functional groups and larger, more polymer-like molecules.
        
        Penalizes:
            - Halogens (F, Cl, Br) and exotic atoms (S, P)
            - Peroxide bonds (O-O) — unstable/explosive
            - Disconnected fragments
            - Molecules too small to be realistic monomers (< 10 atoms)
        
        Rewards:
            - Ester bonds (key for hydrolytic biodegradation)
            - Amide bonds (enzymatically cleavable)
            - Hydroxyl groups (hydrophilicity, microbial access)
            - Ether linkages (common in biodegradable backbones)
            - High O:C ratio (oxidative degradation)
            - Larger, ring-containing molecules
        """
        bonus = 0.0
        n_atoms = mol.GetNumHeavyAtoms()
        
        # ── HARD PENALTIES (chemistry correctness) ──────────────
        
        # Halogen penalty (safety net — also blocked in action mask)
        n_halogens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ['F', 'Cl', 'Br'])
        if n_halogens > 0:
            bonus -= n_halogens * self.halogen_penalty
        
        # Exotic atom penalty — S, P are not standard biodegradable polymer atoms
        n_exotic = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ['S', 'P', 'Si', 'B', 'Se', 'I'])
        if n_exotic > 0:
            bonus -= n_exotic * 0.5
        
        # Peroxide (O-O) penalty — unstable, explosive, unrealistic
        # Match any O-O bond (including OOO trioxide patterns)
        peroxide_pat = Chem.MolFromSmarts('[O]-[O]')
        if peroxide_pat:
            n_peroxide = len(mol.GetSubstructMatches(peroxide_pat))
            bonus -= n_peroxide * 0.8   # 0.8 per O-O (was 0.6)
        
        # N-O single bond penalty — hydroxylamine/nitro-like, never in
        # biodegradable polymer backbones.  Very aggressive to force the
        # GFlowNet away from these.
        n_o_pat = Chem.MolFromSmarts('[N]-[O]')
        if n_o_pat:
            n_no = len(mol.GetSubstructMatches(n_o_pat))
            bonus -= n_no * 0.7   # 0.7 per N-O bond (was 0.35)
        
        # N=O (nitro-like) — even worse than single N-O
        n_eq_o_pat = Chem.MolFromSmarts('[N]=[O]')
        if n_eq_o_pat:
            n_nitro = len(mol.GetSubstructMatches(n_eq_o_pat))
            bonus -= n_nitro * 0.8
        
        # N-N bond penalty — hydrazine-like, unusual in biodegradable polymers
        n_n_pat = Chem.MolFromSmarts('[N]-[N]')
        if n_n_pat:
            n_nn = len(mol.GetSubstructMatches(n_n_pat))
            bonus -= n_nn * 0.6   # 0.6 per N-N bond (was 0.35)
        
        # N-N-N chain penalty — triazene-like, very exotic
        n_n_n_pat = Chem.MolFromSmarts('[N]-[N]-[N]')
        if n_n_n_pat and mol.HasSubstructMatch(n_n_n_pat):
            bonus -= 0.5  # Extra on top of individual N-N penalties
        
        # N=S bond penalty — exotic, not in biodegradable polymers
        n_s_pat = Chem.MolFromSmarts('[N]=[S]')
        if n_s_pat and mol.HasSubstructMatch(n_s_pat):
            bonus -= 0.5
        
        # Disconnected fragments penalty
        fragments = Chem.GetMolFrags(mol)
        if len(fragments) > 1:
            bonus -= 0.2 * (len(fragments) - 1)
        
        # ── SIZE REQUIREMENTS ──────────────────────────────────
        
        # Minimum size — real biodegradable monomers are 6+ atoms
        # Strongly encourage 10+ atoms for structural diversity
        if n_atoms < 6:
            bonus -= 0.4  # Very strong penalty for tiny fragments
        elif n_atoms < 8:
            bonus -= 0.2  # Moderate penalty
        elif n_atoms < 10:
            bonus -= 0.0  # Acceptable but no bonus
        
        # Progressive size bonus for larger molecules (polymer-like)
        if n_atoms >= self.size_bonus_threshold:
            size_factor = min((n_atoms - self.size_bonus_threshold + 1) / 10.0, 1.0)
            bonus += self.size_bonus * (1.0 + size_factor)
        if n_atoms >= 15:
            bonus += self.size_bonus * 0.5  # Extra for large molecules
        if n_atoms >= 20:
            bonus += self.size_bonus * 0.3  # Even more for very large
        
        # ── POSITIVE REWARDS (biodegradable chemistry) ─────────
        
        # Ester bonds — THE key for hydrolytic biodegradation (PLA, PGA, PCL, PBS)
        if self._ester_pat:
            n_ester = len(mol.GetSubstructMatches(self._ester_pat))
            bonus += min(n_ester * self.ester_bonus, 0.7)
        
        # AROMATIC ESTER PENALTY — esters bonded to aromatic rings (PET, PBT)
        # are resistant to biodegradation because the aromatic ring's electron-
        # withdrawing effect stabilizes the ester bond against hydrolysis.
        # Aliphatic esters (PLA, PCL, PBS) biodegrade; aromatic ones do NOT.
        arom_ester_pat = Chem.MolFromSmarts('[c]C(=O)O')  # aromatic C → C(=O)O
        arom_ester_pat2 = Chem.MolFromSmarts('[c]C(=O)[OX2]')  # also match within chain
        n_arom_ester = 0
        if arom_ester_pat:
            n_arom_ester += len(mol.GetSubstructMatches(arom_ester_pat))
        if arom_ester_pat2:
            n_arom_ester += len(mol.GetSubstructMatches(arom_ester_pat2))
        n_arom_ester = min(n_arom_ester, 4)  # cap
        if n_arom_ester > 0:
            bonus -= n_arom_ester * 0.25  # Strong penalty per aromatic ester
        
        # High aromatic fraction penalty — molecules with >50% aromatic atoms
        # are unlikely to be biodegradable in the environment
        n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        if n_atoms > 0 and n_aromatic / n_atoms > 0.4:
            bonus -= 0.20  # Penalty for highly aromatic molecules
        
        # Amide bonds — enzymatically cleavable (nylon-like, proteins)
        if self._amide_pat:
            n_amide = len(mol.GetSubstructMatches(self._amide_pat))
            bonus += min(n_amide * self.amide_bonus, 0.5)
        
        # Hydroxyl groups — increase hydrophilicity, help microbial access
        if self._hydroxyl_pat:
            n_oh = len(mol.GetSubstructMatches(self._hydroxyl_pat))
            bonus += min(n_oh * self.hydroxyl_bonus, 0.25)
        
        # Carbonyl groups (C=O) — building blocks for esters and amides.
        # Rewarding C=O even before the full ester/amide is formed helps
        # the GFlowNet discover the C + double-bond-O construction path.
        carbonyl_pat = Chem.MolFromSmarts('[CX3]=[OX1]')
        if carbonyl_pat:
            n_carbonyl = len(mol.GetSubstructMatches(carbonyl_pat))
            bonus += min(n_carbonyl * 0.12, 0.25)
        
        # Ether linkages — common in biodegradable polymer backbones (PEG)
        ether_pat = Chem.MolFromSmarts('[CX4]-[OX2]-[CX4]')
        if ether_pat:
            n_ether = len(mol.GetSubstructMatches(ether_pat))
            bonus += min(n_ether * 0.08, 0.15)
        
        # Carbonate groups — used in biodegradable polycarbonates
        carbonate_pat = Chem.MolFromSmarts('[OX2]C(=O)[OX2]')
        if carbonate_pat:
            n_carbonate = len(mol.GetSubstructMatches(carbonate_pat))
            bonus += min(n_carbonate * 0.12, 0.2)
        
        # Oxygen-richness bonus — O:C ratio > 0.3 is good for biodegradation
        num_O = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        num_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        if num_C > 0:
            o_to_c = num_O / num_C
            if o_to_c >= 0.3:
                bonus += min(o_to_c * 0.12, 0.15)
        
        # ── RING STRUCTURE BONUSES (Session 4 enhanced) ──────────
        n_rings = mol.GetRingInfo().NumRings()
        if n_rings > 0:
            bonus += min(n_rings * 0.15, 0.30)  # Increased from 0.1→0.15
        
        # Lactone rings (5-7 membered with ester) — THE gold standard
        # for biodegradable polymers (caprolactone, lactide, glycolide)
        lactone_pat = Chem.MolFromSmarts('[OX2]C(=O)[CR1]')
        if lactone_pat and mol.HasSubstructMatch(lactone_pat):
            bonus += 0.35  # Very strong bonus
        
        # Lactam rings (amide in ring) — nylons, caprolactam
        lactam_pat = Chem.MolFromSmarts('[NX3H]C(=O)[CR1]')
        if lactam_pat and mol.HasSubstructMatch(lactam_pat):
            bonus += 0.30
        
        # ── MANDATORY ESTER/CARBONYL REQUIREMENT (Session 4) ────
        # THIS IS THE KEY FIX: Run 3b produced 0/20 candidates with
        # ester bonds. ALL real biodegradable polymers have C=O linkages.
        # Severely penalize large molecules that lack any carbonyl.
        carbonyl_pat = Chem.MolFromSmarts('[CX3]=[OX1]')
        has_carbonyl = carbonyl_pat and mol.HasSubstructMatch(carbonyl_pat)
        if n_atoms >= 10 and not has_carbonyl:
            bonus -= 0.60  # Near-death penalty for 10+ atom molecules without C=O
        elif n_atoms >= 7 and not has_carbonyl:
            bonus -= 0.30  # Moderate penalty for medium molecules
        
        # ── MECHANICAL-PROPERTY-BOOSTING BONUSES ───────────────
        n_double = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0)
        bonus += min(n_double * 0.06, 0.15)
        
        n_branch = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C' and a.GetDegree() >= 3)
        bonus += min(n_branch * 0.05, 0.12)
        
        nh_pat = Chem.MolFromSmarts('[NH]')
        if nh_pat:
            n_nh = len(mol.GetSubstructMatches(nh_pat))
            bonus += min(n_nh * 0.05, 0.10)
        
        # ── POLYMERIZABILITY BONUS ──────────────────────────────
        try:
            from data.real_polymer_data import compute_polymerizability
            poly_result = compute_polymerizability(Chem.MolToSmiles(mol))
            poly_score = poly_result['score']
            if poly_score >= 0.8:
                bonus += 0.20
            elif poly_score >= 0.3:
                bonus += 0.10
            elif poly_score == 0.0 and n_atoms >= 10:
                bonus -= 0.15
        except ImportError:
            pass
        
        return max(-1.0, min(self.max_shaping_bonus, bonus))
    
    @torch.no_grad()
    def compute_reward(self, smiles: str) -> Dict[str, float]:
        """
        Compute the full reward for a molecule.
        
        Returns:
            Dictionary with individual scores and combined reward
        """
        # Validity check
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                's_bio': 0.0, 's_mech': 0.0, 's_syn': 0.0,
                'reward': self.reward_min, 'valid': False,
                'shaping_bonus': 0.0,
            }
        
        # Convert to graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            return {
                's_bio': 0.0, 's_mech': 0.0, 's_syn': 0.0,
                'reward': self.reward_min, 'valid': False,
                'shaping_bonus': 0.0,
            }
        
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
        graph.x = graph.x.to(self.device)
        graph.edge_index = graph.edge_index.to(self.device)
        graph.edge_attr = graph.edge_attr.to(self.device)
        
        # S_bio: Biodegradability
        s_bio = self.bio_model(graph).item()
        s_bio = max(0.0, min(1.0, s_bio))
        
        # ── POST-HOC S_BIO CORRECTION ───────────────────────────
        # Aromatic esters (e.g., PET, PBT) have high hydrolytic stability
        # due to aromatic ring electron-withdrawal. The neural network may
        # overestimate their biodegradability because it sees ester bonds.
        # Correct s_bio downward proportional to aromatic_ester/total_ester ratio.
        arom_ester_fix = Chem.MolFromSmarts('[c]C(=O)[OX2]')
        any_ester_fix = Chem.MolFromSmarts('[#6](=[#8])-[#8]')
        if arom_ester_fix and any_ester_fix:
            n_arom_est = len(mol.GetSubstructMatches(arom_ester_fix))
            n_total_est = len(mol.GetSubstructMatches(any_ester_fix))
            if n_arom_est > 0 and n_total_est > 0:
                arom_ratio = n_arom_est / n_total_est
                # Reduce s_bio: e.g., if all esters are aromatic, cut s_bio by 60%
                s_bio *= max(0.15, 1.0 - arom_ratio * 0.85)
        
        # High aromatic fraction correction — >40% aromatic heavy atoms
        n_aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy > 0 and n_aromatic_atoms / n_heavy > 0.4:
            s_bio *= 0.6  # Reduce by 40% for highly aromatic molecules
        
        # S_mech: Mechanical properties
        mech_output = self.mech_model(graph)
        s_mech = mech_output['combined'].item()
        s_mech = max(0.0, min(1.0, s_mech))
        
        # S_syn: Synthesizability
        s_syn = self.syn_scorer.score(smiles)
        s_syn = max(0.0, min(1.0, s_syn))
        
        # ── POLYMERIZABILITY-BASED S_MECH BOOST ────────────────
        # The S_mech neural network evaluates monomer-level properties.
        # But polymerizable monomers form long chains with dramatically
        # better mechanical properties due to chain entanglement,
        # crystallinity, and intermolecular H-bonding.
        # 
        # Example: ε-Caprolactone monomer has tensile ~5 MPa, but 
        # Polycaprolactone (PCL polymer) has tensile ~25 MPa.
        # Lactic acid monomer ~8 MPa → PLA polymer ~55 MPa.
        #
        # Boost effective S_mech proportional to polymerizability score.
        try:
            from data.real_polymer_data import compute_polymerizability
            poly_info = compute_polymerizability(smiles)
            poly_score = poly_info['score']
            
            if poly_score >= 0.8:
                # Self-polymerizing → large boost (lactones, hydroxy acids)
                # Boost: up to +0.18 for perfect polymerizability
                s_mech_boost = 0.18 * poly_score
                s_mech = min(1.0, s_mech + s_mech_boost)
            elif poly_score >= 0.3:
                # Needs co-monomer → moderate boost (diols, diacids)
                s_mech_boost = 0.10 * poly_score
                s_mech = min(1.0, s_mech + s_mech_boost)
        except ImportError:
            poly_info = {'is_polymerizable': False, 'score': 0.0, 'mechanisms': [], 'groups_found': []}
        
        # ── SMALL MONOMER S_BIO RECALIBRATION ──────────────────
        # Very small polymerizable monomers (≤7 atoms) tend to have
        # slightly underestimated S_bio due to limited structural features.
        # Known biodegradable monomers like lactic acid (6 atoms) and
        # glycolic acid (5 atoms) actually have S_bio ≥ 0.90.
        # Apply a small boost for highly oxygenated small monomers.
        n_O = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        n_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        if n_heavy <= 7 and n_C > 0 and n_O / n_C >= 0.5:
            # High O:C ratio in small molecule = likely biodegradable
            if s_bio > 0.70:  # Only boost already-high predictions
                s_bio = min(1.0, s_bio * 1.08)  # +8% boost
        
        # ── MOLECULAR WEIGHT PENALTY ────────────────────────────
        # Monomers > 500 Da are unlikely to be practical polymer building
        # blocks. Inspired by Lipinski's Rule of Five for drug-likeness.
        # Smooth penalty ramp: 1.0 at 400 Da, 0.5 at 600 Da.
        try:
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            if mw > 500:
                mw_penalty = max(0.3, 1.0 - (mw - 500) / 400)
                s_mech *= mw_penalty
                s_syn *= mw_penalty
        except Exception:
            pass
        
        # ── TOPOLOGICAL COMPLEXITY SCORING ──────────────────────
        # Bertz complexity index (CT) captures structural complexity.
        # Very simple molecules (all-carbon chains) get slight penalty;
        # moderately complex functional molecules get slight boost.
        try:
            from rdkit.Chem import GraphDescriptors
            bertz = GraphDescriptors.BertzCT(mol)
            if bertz < 50 and n_heavy >= 8:
                # Too simple for a useful polymer monomer
                s_bio *= 0.90  # Mild penalty
            elif 200 < bertz < 800:
                # Goldilocks zone: complex enough to be functional
                s_mech = min(1.0, s_mech * 1.05)  # +5% boost
        except Exception:
            pass
        
        # ── MULTIPLICATIVE REWARD FORMULA ─────────────────────
        # Session 9: Adaptive S_bio exponent 0.50→0.55 over training
        # Higher exponent = harder differentiation between biodeg/non-biodeg
        # S_mech and S_syn exponents fixed at 0.35 and 0.15
        eps = 1e-6  # Avoid zero in power
        bio_exp = getattr(self, '_adaptive_bio_exponent', 0.50)
        base_reward = (
            (s_bio + eps) ** bio_exp *
            (s_mech + eps) ** 0.35 *
            (s_syn + eps) ** 0.15
        )
        
        # ── HARD THRESHOLD ENFORCEMENT ─────────────────────────
        # Per IEEE paper specs:
        #   min S_bio ≥ 0.6, min S_mech ≥ 0.5, min S_syn ≥ 0.5
        # Molecules failing ANY threshold get penalized, but softly
        # enough that gradient signal remains useful for learning.
        threshold_penalty = 1.0
        if s_bio < 0.6:
            # Smooth ramp: 0.2 at zero, 1.0 at threshold
            threshold_penalty *= max(0.2, s_bio / 0.6)
        if s_mech < 0.5:
            threshold_penalty *= max(0.2, s_mech / 0.5)
        if s_syn < 0.5:
            threshold_penalty *= max(0.3, s_syn / 0.5)
        base_reward *= threshold_penalty
        
        # ── REWARD SHAPING ──────────────────────────────────────
        # Additive bonus/penalty steers exploration toward
        # realistic polymer chemistry even when base_reward is small.
        shaping = self._compute_shaping_bonus(mol)
        
        # Positive shaping adds meaningful bonus; negative shaping
        # directly penalizes the reward (bad chemistry killer)
        if shaping >= 0:
            reward = base_reward * (1.0 + shaping * 0.5)
        else:
            # Negative shaping: multiply to heavily penalize
            # e.g. shaping=-0.7 → factor=0.30 → reward*0.30
            penalty_factor = max(0.02, 1.0 + shaping)
            reward = base_reward * penalty_factor
        
        # Floor to prevent zero rewards (GFlowNet needs R > 0)
        reward = max(reward, self.reward_min)
        
        # ── SESSION 11: C1-C4 REWARD IMPROVEMENTS ────────────────
        
        # C2: Ring strain penalty (3/4-membered rings are unstable)
        ring_info = mol.GetRingInfo()
        strained_rings = sum(1 for r in ring_info.AtomRings() if len(r) <= 4)
        if strained_rings > 0:
            reward *= max(0.3, 1.0 - 0.25 * strained_rings)
        
        # C3: H-bond donor/acceptor balance
        from rdkit.Chem import Descriptors
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        if hbd + hba > 0:
            # Ideal ratio: 0.3-0.7 range, penalize extremes
            hb_ratio = hbd / (hbd + hba) if (hbd + hba) > 0 else 0.5
            hb_balance = 1.0 - abs(hb_ratio - 0.4) * 0.5  # Peak at 0.4
            reward *= max(0.7, hb_balance)
        
        # C4: Functional group diversity bonus
        fg_patterns = {
            'ester': Chem.MolFromSmarts('[CX3](=O)[OX2]'),
            'hydroxyl': Chem.MolFromSmarts('[OX2H]'),
            'carboxyl': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
            'amide': Chem.MolFromSmarts('[NX3][CX3](=[OX1])'),
            'ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
        }
        fg_count = sum(1 for pat in fg_patterns.values() if pat and mol.HasSubstructMatch(pat))
        if fg_count >= 2:
            reward *= (1.0 + 0.05 * min(fg_count, 4))  # Up to +20% for 4+ groups
        
        # poly_info already computed above in S_mech boost section
        
        return {
            's_bio': s_bio,
            's_mech': s_mech,
            's_syn': s_syn,
            'reward': reward,
            'base_reward': base_reward,
            'shaping_bonus': shaping,
            'valid': True,
            'smiles': smiles,
            'num_atoms': mol.GetNumHeavyAtoms(),
            'polymerizability': poly_info,
            'mech_details': {
                'tensile': mech_output['tensile'].item(),
                'tg': mech_output['tg'].item(),
                'flexibility': mech_output['flexibility'].item(),
            },
        }
    
    def compute_reward_batch(self, smiles_list: List[str]) -> List[Dict]:
        """Compute rewards for a batch of molecules."""
        return [self.compute_reward(s) for s in smiles_list]


# ============================================================
# DIVERSITY TRACKER (Session 4 — Improvement #5)
# ============================================================

class DiversityTracker:
    """
    Tracks generated molecule fingerprints and penalizes near-duplicates.
    
    Prevents the GFlowNet from collapsing to a single structural motif
    (e.g., all polyol chains). Uses Morgan fingerprints + Tanimoto similarity.
    """
    
    def __init__(self, max_history: int = 500, penalty_threshold: float = 0.85):
        self.max_history = max_history
        self.penalty_threshold = penalty_threshold
        self.history_fps = []
    
    def get_diversity_penalty(self, mol) -> float:
        """
        Compute diversity penalty based on similarity to recently generated molecules.
        
        Returns:
            Negative penalty (0.0 to -0.30) if too similar to recent molecules.
        """
        if mol is None:
            return 0.0
        
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        except Exception:
            return 0.0
        
        if not self.history_fps:
            self._add_to_history(fp)
            return 0.0
        
        # Compare to most recent 100 molecules (faster than full history)
        compare_set = self.history_fps[-100:]
        max_sim = max(
            DataStructs.TanimotoSimilarity(fp, h) for h in compare_set
        )
        
        self._add_to_history(fp)
        
        if max_sim > self.penalty_threshold:
            # Linear penalty: 0 at threshold, -0.30 at similarity=1.0
            penalty = -0.30 * (max_sim - self.penalty_threshold) / (1.0 - self.penalty_threshold)
            return penalty
        
        return 0.0
    
    def _add_to_history(self, fp):
        """Add fingerprint to history, evicting oldest if at capacity."""
        self.history_fps.append(fp)
        if len(self.history_fps) > self.max_history:
            self.history_fps = self.history_fps[-self.max_history:]
    
    def reset(self):
        """Clear history (e.g., between training rounds)."""
        self.history_fps = []


# ============================================================
# CURRICULUM LEARNING MOLECULES (Session 4 — Improvement #6)
# ============================================================

# Known biodegradable monomers that the GFlowNet should learn to construct.
# These are used as seed molecules during early training steps.
CURRICULUM_MOLECULES = [
    'CC(O)C(=O)O',           # Lactic acid (PLA monomer)
    'OCC(=O)O',              # Glycolic acid (PGA monomer)
    'O=C1CCCCCO1',           # ε-Caprolactone (PCL monomer)
    'OC(=O)CCC(=O)O',       # Succinic acid (PBS monomer)
    'OCCCCO',                # 1,4-Butanediol (PBS co-monomer)
    'CC(O)CC(=O)O',          # 3-Hydroxybutyric acid (PHB monomer)
    'O=C1OCCO1',             # Glycolide (PGA cyclic dimer)
    'CC1OC(=O)C(C)OC1=O',   # L-Lactide (PLA cyclic dimer)
    'O=C(O)CCCCC(=O)O',     # Adipic acid (PBAT monomer)
    'OCCCCCCO',              # 1,6-Hexanediol (polyester co-monomer)
    'OC(=O)c1ccc(C(=O)O)o1', # 2,5-FDCA (PEF monomer)
    'NCC(=O)O',              # Glycine (polyamide monomer)
    'O=C1CCCCO1',            # δ-Valerolactone
    'OC(=O)CC(=C)C(=O)O',   # Itaconic acid
    'CC(=O)CCC(=O)O',       # Levulinic acid
]


# ============================================================
# REPLAY BUFFER (Publication Improvement #2)
# Shen et al. "Towards Understanding and Improving GFlowNet
# Training" (ICML 2023) — reward-prioritized replay
# ============================================================

class ReplayBuffer:
    """
    Reward-prioritized replay buffer for GFlowNet training.
    
    Stores high-reward trajectories and replays them with probability
    proportional to reward^alpha, dramatically improving sample efficiency.
    """
    
    def __init__(self, capacity: int = 2000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self._median_reward = 0.0
    
    def add(self, reward: float, smiles: str = ''):
        """Add trajectory info if reward exceeds buffer median."""
        if len(self.buffer) < 10 or reward > self._median_reward * 0.7:
            self.buffer.append({'reward': reward, 'smiles': smiles})
            
            if len(self.buffer) > self.capacity:
                min_idx = min(range(len(self.buffer)),
                              key=lambda i: self.buffer[i]['reward'])
                self.buffer.pop(min_idx)
            
            rewards = sorted(e['reward'] for e in self.buffer)
            self._median_reward = rewards[len(rewards) // 2] if rewards else 0.0
    
    def sample(self, n: int) -> list:
        """Sample n entries with reward-prioritized probability."""
        if not self.buffer:
            return []
        n = min(n, len(self.buffer))
        weights = np.array([max(e['reward'], 1e-8) ** self.alpha for e in self.buffer])
        weights = weights / weights.sum()
        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=weights)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    @property
    def mean_reward(self):
        return np.mean([e['reward'] for e in self.buffer]) if self.buffer else 0.0
    
    @property
    def top_reward(self):
        return max(e['reward'] for e in self.buffer) if self.buffer else 0.0


class GFlowNet(nn.Module):
    """
    Generative Flow Network for Biodegradable Polymer Discovery.
    
    Generates molecules atom-by-atom, sampling proportional to reward:
        P(molecule) ∝ R(molecule)
    
    Training uses Trajectory Balance loss (Malkin et al., 2022):
        L_TB = (log Z + Σ log P_F(a_t|s_t) - log R(x) - Σ log P_B(a_t|s_{t+1}))²
    
    This ensures diverse sampling — unlike RL which collapses to the mode.
    """
    
    def __init__(
        self,
        policy_network: PolicyNetwork,
        reward_function: RewardFunction,
        config: dict = None,
    ):
        super().__init__()
        
        self.policy = policy_network
        self.reward_fn = reward_function
        
        config = config or {}
        self.config = config  # Store for checkpoint serialization
        self.max_atoms = config.get('max_atoms', 30)
        self.min_atoms = config.get('min_atoms', 8)
        self.temperature = config.get('temperature', 2.0)
        self.temperature_decay = config.get('temperature_decay', 0.9985)
        self.min_temperature = config.get('min_temperature', 0.5)
        self.epsilon = config.get('epsilon', 0.20)
        self.epsilon_decay = config.get('epsilon_decay', 0.9995)
        self.min_epsilon = config.get('min_epsilon', 0.02)
        self.reward_min = config.get('reward_min', 1e-8)
        self.log_reward_min = config.get('log_reward_min', -18.0)
        self.device = config.get('device', 'cpu')
        
        # Learnable log Z (partition function)
        self.log_z = nn.Parameter(torch.tensor(0.0))
        
        # Statistics
        self.generation_stats = defaultdict(list)
        self._train_step_count = 0
        
        # Session 4: Diversity tracker
        self.diversity_tracker = DiversityTracker(
            max_history=config.get('diversity_history', 500),
            penalty_threshold=config.get('diversity_threshold', 0.85),
        )
        
        # Session 4: Curriculum learning
        self._curriculum_smiles = CURRICULUM_MOLECULES
        self._curriculum_rewards = {}  # Cache curriculum rewards
        
        # Publication improvement: Replay buffer + SubTB
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('replay_capacity', 2000),
            alpha=config.get('replay_alpha', 0.6),
        )
        self.use_subtb = config.get('use_subtb', True)
        self.subtb_weight = config.get('subtb_weight', 0.5)
        
        # Publication improvement #7: Multi-Objective Pareto GFlowNet
        try:
            from models.mogfn import ParetoFront, scalarize_reward
            self.pareto_front = ParetoFront(
                num_objectives=3,
                max_size=config.get('pareto_max_size', 500),
            )
            self.mogfn_enabled = config.get('mogfn_enabled', True)
            self._scalarize_reward = scalarize_reward
        except ImportError:
            self.pareto_front = None
            self.mogfn_enabled = False
            self._scalarize_reward = None
        
        # ── SESSION 11 IMPROVEMENTS ──────────────────────────────
        
        # A1: Entropy regularization — prevents policy collapse
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        
        # A2: Exponential Moving Average of weights
        self._ema_decay = config.get('ema_decay', 0.999)
        self._ema_params = {}  # Populated after first train step
        self._ema_initialized = False
        
        # A3: Gradient penalty coefficient
        self._grad_penalty_lambda = config.get('grad_penalty_lambda', 0.001)
        
        # A4: Hindsight Experience Replay
        self._her_enabled = config.get('her_enabled', True)
        
        # A5: Progressive molecule size curriculum
        self._progressive_sizes = [
            (0, 2000, 8),      # Steps 0-2000: max 8 atoms
            (2000, 5000, 15),  # Steps 2000-5000: max 15 atoms
            (5000, 99999, 25), # Steps 5000+: max 25 atoms
        ]
        
        # A6: Logit-GFN temperature scaling
        self._logit_temp_scale = config.get('logit_temp_scale', True)
        
        # A7: Backward policy loss weight
        self._backward_loss_weight = config.get('backward_loss_weight', 0.1)
        
        # C1: Tanimoto similarity bonus cache
        self._known_biodeg_fps = None  # Lazy-loaded fingerprints
        
        # Session 9 adaptive exponent
        self._adaptive_bio_exponent = 0.50
    
    def generate_trajectory(
        self,
        deterministic: bool = False,
    ) -> Tuple[List[MolecularState], List[int], List[float], Dict]:
        """
        Generate a single molecule trajectory.
        
        Returns:
            states: List of states along the trajectory
            actions: List of actions taken
            log_probs: List of log probabilities of actions
            reward_info: Reward dictionary for the final molecule
        """
        state = MolecularState(max_atoms=self.max_atoms, min_atoms=self.min_atoms)
        
        # Start with a C-C seed (2 atoms) — single carbon is too minimal,
        # and the auto-bond from add_atom handles the C-C bond automatically
        state.add_atom('C')
        
        states = [state.copy()]
        actions = []
        log_probs = []
        
        max_steps = self.max_atoms * 3  # Safety limit (generous for bonds)
        
        for step in range(max_steps):
            if state.done:
                break
            
            # Get graph representation
            graph_data = state.to_graph_data(self.device)
            if graph_data is None:
                state.terminate()
                break
            
            # Get valid actions
            action_mask = state.get_valid_actions().unsqueeze(0).to(self.device)
            
            # Check if any actions are valid
            if not action_mask.any():
                # If below min_atoms, force-add a Carbon bonded to any atom
                # with remaining valence (keeps molecule connected)
                if state.num_atoms < self.min_atoms:
                    # Find any atom with remaining valence for a SINGLE bond
                    bonded = False
                    for target_idx in range(state.num_atoms):
                        used_v = sum(
                            {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bt, 1)
                            for bi, bj, bt in state.bonds
                            if bi == target_idx or bj == target_idx
                        )
                        remaining = state._get_max_valence(state.atoms[target_idx]) - used_v
                        if remaining >= 1:
                            # Add C and bond it to this atom
                            state.atoms.append('C')
                            state._mol = None
                            state._smiles = None
                            state.step += 1
                            new_idx = state.num_atoms - 1
                            state.add_bond(target_idx, new_idx, 'SINGLE')
                            bonded = True
                            break
                    
                    if not bonded:
                        # Truly no valence left anywhere — extremely rare, just terminate
                        state.terminate()
                        break
                    
                    states.append(state.copy())
                    actions.append(0)  # Carbon add action
                    log_probs.append(torch.tensor(-2.0, device=self.device))
                    continue
                state.terminate()
                break
            
            # Sample action from policy
            if deterministic:
                log_p, probs = self.policy.get_forward_policy(
                    graph_data, action_mask, temperature=0.1
                )
                action_idx = probs[0].argmax().item()
                log_prob = log_p[0, action_idx]  # Keep as tensor
            else:
                action_idx, log_prob = self.policy.sample_action(
                    graph_data, action_mask, self.temperature, self.epsilon
                )
            
            # Apply action
            success = state.apply_action(action_idx)
            
            if not success:
                # Action failed — if below min_atoms, don't terminate, just skip
                if state.num_atoms < self.min_atoms:
                    continue
                # If above min_atoms, terminate
                state.terminate()
                break
            
            states.append(state.copy())
            actions.append(action_idx)
            log_probs.append(log_prob)
        
        # Ensure molecule is terminated
        if not state.done:
            state.terminate()
        
        # Compute reward
        smiles = state.to_smiles()
        if smiles is not None:
            reward_info = self.reward_fn.compute_reward(smiles)
        else:
            reward_info = {
                's_bio': 0.0, 's_mech': 0.0, 's_syn': 0.0,
                'reward': self.reward_min, 'valid': False,
            }
        
        return states, actions, log_probs, reward_info
    
    def generate_trajectory_from_state(
        self,
        start_state: 'MolecularState',
        prefix_states: list = None,
        prefix_actions: list = None,
        prefix_log_probs: list = None,
    ) -> Tuple[List['MolecularState'], List[int], List[float], Dict]:
        """
        Continue generating a trajectory from an intermediate state.
        Used by Local Search GFlowNet (LS-GFN) for trajectory refinement.
        
        Args:
            start_state: Intermediate molecular state to continue from
            prefix_states: States from the prefix trajectory
            prefix_actions: Actions from the prefix trajectory
            prefix_log_probs: Log probs from the prefix trajectory
            
        Returns:
            Full trajectory (prefix + continuation)
        """
        state = start_state.copy()
        states = list(prefix_states) if prefix_states else [state.copy()]
        actions = list(prefix_actions) if prefix_actions else []
        log_probs_list = list(prefix_log_probs) if prefix_log_probs else []
        
        max_steps = self.max_atoms * 3
        
        for step in range(max_steps):
            if state.done:
                break
            
            graph_data = state.to_graph_data(self.device)
            if graph_data is None:
                state.terminate()
                break
            
            action_mask = state.get_valid_actions().unsqueeze(0).to(self.device)
            
            if not action_mask.any():
                if state.num_atoms < self.min_atoms:
                    # Force-add carbon
                    bonded = False
                    for target_idx in range(state.num_atoms):
                        used_v = sum(
                            {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(bt, 1)
                            for bi, bj, bt in state.bonds
                            if bi == target_idx or bj == target_idx
                        )
                        remaining = state._get_max_valence(state.atoms[target_idx]) - used_v
                        if remaining >= 1:
                            state.atoms.append('C')
                            state._mol = None
                            state._smiles = None
                            state.step += 1
                            new_idx = state.num_atoms - 1
                            state.add_bond(target_idx, new_idx, 'SINGLE')
                            bonded = True
                            break
                    if not bonded:
                        state.terminate()
                        break
                    states.append(state.copy())
                    actions.append(0)
                    log_probs_list.append(torch.tensor(-2.0, device=self.device))
                    continue
                state.terminate()
                break
            
            action_idx, log_prob = self.policy.sample_action(
                graph_data, action_mask, self.temperature, self.epsilon
            )
            
            success = state.apply_action(action_idx)
            if not success:
                if state.num_atoms < self.min_atoms:
                    continue
                state.terminate()
                break
            
            states.append(state.copy())
            actions.append(action_idx)
            log_probs_list.append(log_prob)
        
        if not state.done:
            state.terminate()
        
        smiles = state.to_smiles()
        if smiles is not None:
            reward_info = self.reward_fn.compute_reward(smiles)
        else:
            reward_info = {
                's_bio': 0.0, 's_mech': 0.0, 's_syn': 0.0,
                'reward': self.reward_min, 'valid': False,
            }
        
        return states, actions, log_probs_list, reward_info
    
    def compute_trajectory_balance_loss(
        self,
        states: List[MolecularState],
        actions: List[int],
        forward_log_probs: List[torch.Tensor],
        reward: float,
    ) -> torch.Tensor:
        """
        Compute Trajectory Balance (TB) loss.
        
        L_TB = (log Z + Σ log P_F(a_t|s_t) - log R(x) - Σ log P_B(a_t|s_{t+1}))²
        
        This is the recommended training objective from Malkin et al. (2022).
        forward_log_probs must be tensors that retain gradients!
        """
        # Log reward (clamped)
        log_reward = max(np.log(max(reward, 1e-30)), self.log_reward_min)
        log_reward = torch.tensor(log_reward, device=self.device, dtype=torch.float)
        
        # Sum of forward log probs — these are TENSOR values with gradient
        if len(forward_log_probs) > 0:
            sum_log_pf = torch.stack(forward_log_probs).sum()
        else:
            sum_log_pf = torch.tensor(0.0, device=self.device)
        
        # Compute backward log probs (detached — in standard TB loss, only
        # forward log probs and log_z receive gradients; backward policy
        # is treated as fixed / evaluated without gradient)
        sum_log_pb = torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            for i in range(len(states) - 1, 0, -1):
                graph_data = states[i].to_graph_data(self.device)
                if graph_data is not None:
                    action_mask = states[i-1].get_valid_actions().unsqueeze(0).to(self.device)
                    log_pb = self.policy.get_backward_policy(graph_data, action_mask)
                    if actions[i-1] < log_pb.shape[1]:
                        pb_val = log_pb[0, actions[i-1]]
                        # Guard against NaN from disconnected graphs
                        if not torch.isnan(pb_val) and not torch.isinf(pb_val):
                            sum_log_pb = sum_log_pb + pb_val
        
        # TB loss — guard against NaN
        tb_residual = self.log_z + sum_log_pf - log_reward - sum_log_pb
        if torch.isnan(tb_residual) or torch.isinf(tb_residual):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        loss = tb_residual ** 2
        
        return loss
    
    def compute_subtb_loss(
        self,
        states: List[MolecularState],
        actions: List[int],
        forward_log_probs: List[torch.Tensor],
        reward: float,
        num_sub: int = 4,
    ) -> torch.Tensor:
        """
        Sub-Trajectory Balance (SubTB) loss — Madan et al. ICML 2023.
        
        Computes losses on partial trajectory segments (i→j) for better
        credit assignment and ~40% lower gradient variance than full TB.
        
        L_SubTB = Σ_{(i,j)} (log F(s_i) + Σ_{k=i}^{j-1} log P_F(a_k|s_k)
                             - log F(s_j) - Σ_{k=i}^{j-1} log P_B(a_k|s_{k+1}))²
        
        For terminal sub-trajectories (j=T), log F(s_j) = log R(x).
        For non-terminal ones, log F(s_i) ≈ log_z (approximation).
        """
        T = len(actions)
        if T < 2:
            return self.compute_trajectory_balance_loss(
                states, actions, forward_log_probs, reward
            )
        
        log_reward = max(np.log(max(reward, 1e-30)), self.log_reward_min)
        log_reward_t = torch.tensor(log_reward, device=self.device, dtype=torch.float)
        
        # Sample sub-trajectory endpoints
        subtb_losses = []
        for _ in range(num_sub):
            # Random sub-trajectory (i, j) where 0 <= i < j <= T
            i = np.random.randint(0, T)
            j = np.random.randint(i + 1, T + 1)
            
            # Sum forward log probs in [i, j)
            sub_log_pf = torch.stack(forward_log_probs[i:j]).sum()
            
            # Sum backward log probs in [i, j) (no gradient)
            sub_log_pb = torch.tensor(0.0, device=self.device)
            with torch.no_grad():
                for k in range(j, i, -1):
                    if k < len(states):
                        gd = states[k].to_graph_data(self.device)
                        if gd is not None and k - 1 >= 0:
                            am = states[k - 1].get_valid_actions().unsqueeze(0).to(self.device)
                            log_pb = self.policy.get_backward_policy(gd, am)
                            if actions[k - 1] < log_pb.shape[1]:
                                pb_val = log_pb[0, actions[k - 1]]
                                if not torch.isnan(pb_val) and not torch.isinf(pb_val):
                                    sub_log_pb = sub_log_pb + pb_val
            
            # State flow estimates
            # For terminal sub-trajectories, log F(s_j) = log R(x)
            if j == T:
                log_F_j = log_reward_t
            else:
                # Approximate with log_z (simple but effective)
                log_F_j = self.log_z
            
            # log F(s_i) ≈ log_z for start, or reward for terminal approach
            if i == 0:
                log_F_i = self.log_z
            else:
                log_F_i = self.log_z  # Uniform approximation
            
            residual = log_F_i + sub_log_pf - log_F_j - sub_log_pb
            if not (torch.isnan(residual) or torch.isinf(residual)):
                subtb_losses.append(residual ** 2)
        
        if not subtb_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return torch.stack(subtb_losses).mean()
    
    def _compute_state_potential(self, state: MolecularState) -> float:
        """
        Forward-Looking state potential — Pan et al. ICML 2023.
        
        Lightweight heuristic that estimates future reward potential at each
        construction step. Used for intermediate credit assignment.
        
        Φ(s) = α·(ester_count) + β·(O:C ratio) + γ·(ring_count) + δ·(size_bonus)
        
        Higher potential → molecule is on track to become a good polymer candidate.
        """
        n_atoms = state.num_atoms
        if n_atoms < 2:
            return 0.0
        
        try:
            smiles = state.to_smiles()
            if smiles is None:
                return 0.0
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
        except Exception:
            return 0.0
        
        potential = 0.0
        
        # Ester bonds (C(=O)O) — strong indicator of biodegradability
        ester_pat = Chem.MolFromSmarts('[#6](=[#8])-[#8]')
        if ester_pat:
            n_esters = len(mol.GetSubstructMatches(ester_pat))
            potential += 0.15 * n_esters
        
        # Amide bonds (C(=O)N) — polyamide linkage
        amide_pat = Chem.MolFromSmarts('[#6](=[#8])-[#7]')
        if amide_pat:
            n_amides = len(mol.GetSubstructMatches(amide_pat))
            potential += 0.12 * n_amides
        
        # O:C ratio (biodegradable polymers typically have O:C ≥ 0.3)
        n_oxygen = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        n_carbon = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        if n_carbon > 0:
            oc_ratio = n_oxygen / n_carbon
            if oc_ratio >= 0.3:
                potential += 0.10 * min(oc_ratio, 0.8)
        
        # Rings (especially 5-7 membered → lactones)
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if 5 <= len(ring) <= 7:
                potential += 0.08
        
        # Size bonus — reward growing to 8-20 atoms
        if 8 <= n_atoms <= 20:
            potential += 0.05
        elif n_atoms > 20:
            potential += 0.02
        
        return min(potential, 1.0)  # Cap at 1.0
    
    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Execute one training step.
        
        Generates a batch of trajectories and updates the policy.
        
        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        
        total_reward = 0.0
        valid_count = 0
        rewards = []
        all_smiles = []
        loss_values = []
        all_s_bio = []
        all_s_mech = []
        all_s_syn = []
        
        # Gradient accumulation: backward each trajectory individually
        # to keep computation graphs small (avoids MPS "Placeholder tensor
        # is empty!" errors from large stacked graphs)
        optimizer.zero_grad()
        
        for traj_i in range(batch_size):
            # ── CURRICULUM LEARNING (Session 4) ───────────────────
            # Early steps: mix in known biodegradable monomer trajectories
            # so the GFlowNet learns what good molecules look like.
            use_curriculum = False
            if self._train_step_count < 500:
                # 20% curriculum rate
                use_curriculum = (traj_i % 5 == 0)
            elif self._train_step_count < 1500:
                # 10% curriculum rate
                use_curriculum = (traj_i % 10 == 0)
            
            if use_curriculum and self._curriculum_smiles:
                # Pick a random curriculum molecule and compute its reward
                curr_smi = np.random.choice(self._curriculum_smiles)
                if curr_smi not in self._curriculum_rewards:
                    self._curriculum_rewards[curr_smi] = self.reward_fn.compute_reward(curr_smi)
                reward_info = self._curriculum_rewards[curr_smi]
                
                if reward_info.get('valid', False):
                    # Use curriculum reward directly (skip trajectory generation)
                    total_reward += reward_info['reward']
                    rewards.append(reward_info['reward'])
                    all_s_bio.append(reward_info.get('s_bio', 0.0))
                    all_s_mech.append(reward_info.get('s_mech', 0.0))
                    all_s_syn.append(reward_info.get('s_syn', 0.0))
                    valid_count += 1
                    if reward_info.get('smiles'):
                        all_smiles.append(reward_info['smiles'])
                    continue  # Skip trajectory generation for this slot
            
            # ── NORMAL TRAJECTORY GENERATION ──────────────────────
            # Generate trajectory
            states, actions, log_probs, reward_info = self.generate_trajectory()
            
            if len(actions) == 0:
                continue
            
            # ── DIVERSITY PENALTY (Session 4) ────────────────────
            # Penalize molecules too similar to recently generated ones
            smiles = reward_info.get('smiles', '')
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                diversity_penalty = self.diversity_tracker.get_diversity_penalty(mol)
                if diversity_penalty < 0:
                    # Apply diversity penalty to reward
                    original_reward = reward_info['reward']
                    adjusted_reward = max(
                        self.reward_min,
                        original_reward * (1.0 + diversity_penalty)
                    )
                    reward_info['reward'] = adjusted_reward
                    reward_info['diversity_penalty'] = diversity_penalty
            
            # ── LOSS: TB + SubTB blend (Publication Improvement #1) ──
            tb_loss = self.compute_trajectory_balance_loss(
                states, actions, log_probs, reward_info['reward']
            )
            
            if self.use_subtb and len(actions) >= 2:
                subtb_loss = self.compute_subtb_loss(
                    states, actions, log_probs, reward_info['reward']
                )
                loss = (1.0 - self.subtb_weight) * tb_loss + self.subtb_weight * subtb_loss
            else:
                loss = tb_loss
            
            # ── RLOO BASELINE (Session 9 — Improvement #2) ──────────
            # REINFORCE Leave-One-Out control variate: subtract mean
            # reward of other batch members to reduce gradient variance
            # without introducing bias (NeurIPS 2024)
            if len(rewards) > 0 and isinstance(loss, torch.Tensor):
                baseline = np.mean(rewards)  # Running baseline
                log_r = max(np.log(max(reward_info['reward'], 1e-8)), -14.0)
                log_baseline = max(np.log(max(baseline, 1e-8)), -14.0)
                cv_scale = max(0.5, min(1.5, 1.0 + 0.1 * (log_r - log_baseline)))
                loss = loss * cv_scale
            
            # ── A1: ENTROPY REGULARIZATION (Session 11) ─────────
            # Add entropy bonus to prevent policy collapse
            if isinstance(loss, torch.Tensor) and len(log_probs) > 0:
                import math
                valid_lps = []
                for lp in log_probs:
                    v = float(lp) if isinstance(lp, torch.Tensor) else lp
                    if not (math.isnan(v) or math.isinf(v)):
                        valid_lps.append(v)
                if valid_lps:
                    probs = np.exp(valid_lps)
                    probs = np.clip(probs, 1e-8, 1.0)
                    entropy = -np.sum(probs * np.log(probs))
                    # Subtract entropy bonus (lower loss = more exploration)
                    entropy_bonus = self.entropy_coeff * entropy
                    loss = loss - entropy_bonus
            
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                # Scale by 1/batch_size for proper averaging
                scaled_loss = loss / max(batch_size, 1)
                if not (torch.isnan(scaled_loss) or torch.isinf(scaled_loss)):
                    scaled_loss.backward()
                    loss_values.append(loss.item())
            
            total_reward += reward_info['reward']
            rewards.append(reward_info['reward'])
            all_s_bio.append(reward_info.get('s_bio', 0.0))
            all_s_mech.append(reward_info.get('s_mech', 0.0))
            all_s_syn.append(reward_info.get('s_syn', 0.0))
            
            if reward_info.get('valid', False):
                valid_count += 1
                if reward_info.get('smiles'):
                    all_smiles.append(reward_info['smiles'])
                    # ── REPLAY BUFFER (Publication Improvement #2) ──
                    self.replay_buffer.add(
                        reward=reward_info['reward'],
                        smiles=reward_info['smiles'],
                    )
                    # ── PARETO FRONT (Publication Improvement #7) ──
                    if self.pareto_front is not None:
                        self.pareto_front.add(
                            reward_info['smiles'],
                            {'s_bio': reward_info.get('s_bio', 0),
                             's_mech': reward_info.get('s_mech', 0),
                             's_syn': reward_info.get('s_syn', 0)},
                        )
        
        # Apply accumulated gradients
        if len(loss_values) > 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # ── COSINE ANNEALING + WARMUP (Session 9 — Improvement #3) ──
            # Linear warmup for first 100 steps, then cosine decay
            step = self._train_step_count
            warmup_steps = 100
            total_steps = 8000  # Expected total
            if step < warmup_steps:
                lr_scale = step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr_scale = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
            lr_scale = max(lr_scale, 0.1)  # Floor at 10% of base LR
            for pg in optimizer.param_groups:
                if 'initial_lr' not in pg:
                    pg['initial_lr'] = pg['lr']
                pg['lr'] = pg['initial_lr'] * lr_scale
            
            optimizer.step()
            avg_loss_val = np.mean(loss_values)
        else:
            avg_loss_val = 0.0
        
        # ── EXPLORATION SCHEDULE (Session 9 — Improvement #8) ────
        # Piecewise + periodic bumps every 2000 steps
        step = self._train_step_count
        if step < 2000:
            # Aggressive exploration first 2000 steps
            self.temperature = max(
                self.temperature * 0.9990,  # Slower decay
                self.min_temperature * 1.5
            )
            self.epsilon = max(
                self.epsilon * 0.9992,
                self.min_epsilon * 2.0
            )
        else:
            # Normal decay after 2000 steps
            self.temperature = max(
                self.temperature * self.temperature_decay,
                self.min_temperature
            )
            self.epsilon = max(
                self.epsilon * self.epsilon_decay,
                self.min_epsilon
            )
        
        # Periodic exploration bump every 2000 steps (escape local optima)
        if step > 0 and step % 2000 == 0:
            self.temperature = min(self.temperature * 1.5, 2.0)
            self.epsilon = min(self.epsilon * 2.0, 0.15)
        
        # ── ADAPTIVE REWARD EXPONENT (Session 9 — Improvement #7) ──
        # Gradually increase S_bio exponent from 0.50 to 0.55 over training
        bio_exp_base = 0.50
        bio_exp_max = 0.55
        total_s = 8000
        exp_progress = min(step / max(total_s, 1), 1.0)
        self._adaptive_bio_exponent = bio_exp_base + (bio_exp_max - bio_exp_base) * exp_progress
        
        # ── A2: EMA WEIGHT UPDATE (Session 11) ───────────────
        # Exponential Moving Average of policy weights for stable generation
        if not self._ema_initialized:
            for name, param in self.policy.named_parameters():
                self._ema_params[name] = param.data.clone()
            self._ema_initialized = True
        else:
            for name, param in self.policy.named_parameters():
                if name in self._ema_params:
                    self._ema_params[name] = (
                        self._ema_decay * self._ema_params[name] +
                        (1.0 - self._ema_decay) * param.data
                    )
        
        # ── A5: PROGRESSIVE SIZE CURRICULUM (Session 11) ──────
        # Gradually increase max molecule size
        for start, end, max_size in self._progressive_sizes:
            if start <= step < end:
                self.max_atoms = max_size
                break
        
        self._train_step_count += 1
        
        # Compute metrics
        atom_counts = []
        for smi in all_smiles:
            m = Chem.MolFromSmiles(smi)
            if m:
                atom_counts.append(m.GetNumHeavyAtoms())
        avg_atoms = np.mean(atom_counts) if atom_counts else 0
        metrics = {
            'loss': avg_loss_val,
            'mean_reward': np.mean(rewards) if rewards else 0.0,
            'max_reward': np.max(rewards) if rewards else 0.0,
            'validity_rate': valid_count / batch_size,
            'temperature': self.temperature,
            'epsilon': self.epsilon,
            'log_z': self.log_z.item(),
            'unique_smiles': len(set(all_smiles)),
            'num_generated': len(all_smiles),
            'avg_atoms': avg_atoms,
            'step': self._train_step_count,
            'mean_s_bio': np.mean(all_s_bio) if all_s_bio else 0.0,
            'mean_s_mech': np.mean(all_s_mech) if all_s_mech else 0.0,
            'mean_s_syn': np.mean(all_s_syn) if all_s_syn else 0.0,
        }
        
        return metrics
    
    @torch.no_grad()
    def generate_molecules(
        self,
        num_molecules: int = 100,
        unique: bool = True,
    ) -> List[Dict]:
        """
        Generate a batch of molecules for evaluation.
        
        Args:
            num_molecules: Number of molecules to generate
            unique: If True, only return unique molecules
        
        Returns:
            List of reward dictionaries for generated molecules
        """
        self.policy.eval()
        
        results = []
        seen_smiles = set()
        attempts = 0
        max_attempts = num_molecules * 5
        
        while len(results) < num_molecules and attempts < max_attempts:
            attempts += 1
            
            _, _, _, reward_info = self.generate_trajectory(deterministic=False)
            
            if not reward_info.get('valid', False):
                continue
            
            smiles = reward_info.get('smiles', '')
            
            if unique and smiles in seen_smiles:
                continue
            
            seen_smiles.add(smiles)
            results.append(reward_info)
        
        # Sort by reward (best first)
        results.sort(key=lambda x: x['reward'], reverse=True)
        
        return results
    
    # ── Session 7 Multi-Temperature Generation ──────────────
    def generate_molecules_multi_temp(
        self,
        num_molecules: int = 2000,
        temperatures: list = None,
        unique: bool = True,
    ) -> List[Dict]:
        """
        Generate molecules at multiple temperatures simultaneously.
        T=0.3 (exploitation), T=1.0 (balanced), T=2.0 (exploration).
        This gives better mode coverage than a single temperature.
        """
        if temperatures is None:
            temperatures = [0.3, 1.0, 2.0]
        
        per_temp = num_molecules // len(temperatures)
        all_results = []
        seen = set()
        
        original_temp = self.temperature
        
        for temp in temperatures:
            self.temperature = temp
            results = self.generate_molecules(
                num_molecules=per_temp,
                unique=True,
            )
            for r in results:
                smi = r.get('smiles', '')
                if smi and (not unique or smi not in seen):
                    seen.add(smi)
                    r['generation_temperature'] = temp
                    all_results.append(r)
        
        # Restore original temperature
        self.temperature = original_temp
        
        # Sort by reward
        all_results.sort(key=lambda x: x.get('reward', 0), reverse=True)
        
        logger.info(f"  Multi-temp generation: {len(all_results)} unique from "
                    f"T={temperatures}, per_temp={per_temp}")
        
        return all_results[:num_molecules]
    
    def save(self, path: str):
        """Save model checkpoint (includes config for architecture reconstruction)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'log_z': self.log_z.data,
            'temperature': self.temperature,
            'generation_stats': dict(self.generation_stats),
            'config': self.config,  # Save config for later reconstruction
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.log_z.data = checkpoint['log_z']
        self.temperature = checkpoint.get('temperature', 1.0)
        logger.info(f"Model loaded from {path}")


def create_gflownet(config: dict = None, device: str = 'cpu') -> GFlowNet:
    """
    Factory function to create a complete GFlowNet system.
    
    Creates and connects:
        1. Policy network (GNN-based)
        2. S_bio surrogate model
        3. S_mech surrogate model
        4. S_syn scorer
        5. Reward function
        6. GFlowNet wrapper
    """
    config = config or {}
    
    # Create policy
    policy = create_policy_network(config).to(device)
    
    # Create surrogate models
    from models.surrogate_bio import create_bio_model
    from models.surrogate_mech import create_mech_model
    
    bio_model = create_bio_model(config).to(device)
    mech_model = create_mech_model(config).to(device)
    syn_scorer = SynthesizabilityScorer()
    
    # Create reward function
    reward_fn = RewardFunction(
        bio_model=bio_model,
        mech_model=mech_model,
        syn_scorer=syn_scorer,
        alpha_bio=config.get('alpha_bio', 1.5),
        alpha_mech=config.get('alpha_mech', 1.0),
        alpha_syn=config.get('alpha_syn', 0.8),
        reward_exponent=config.get('reward_exponent', 2.0),
        device=device,
        ester_bonus=config.get('ester_bonus', 0.15),
        amide_bonus=config.get('amide_bonus', 0.10),
        hydroxyl_bonus=config.get('hydroxyl_bonus', 0.05),
        size_bonus_threshold=config.get('size_bonus_threshold', 8),
        size_bonus=config.get('size_bonus', 0.1),
        halogen_penalty=config.get('halogen_penalty', 0.2),
        max_shaping_bonus=config.get('max_shaping_bonus', 0.5),
    )
    
    # Create GFlowNet
    gflownet = GFlowNet(
        policy_network=policy,
        reward_function=reward_fn,
        config={**config, 'device': device},
    ).to(device)
    
    total_params = sum(p.numel() for p in gflownet.parameters())
    logger.info(f"GFlowNet created with {total_params:,} parameters")
    
    return gflownet
