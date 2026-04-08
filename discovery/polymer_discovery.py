"""
Polymer Discovery Engine
=========================
Input-driven polymer discovery: given a target polymer (e.g. "polyethylene"),
the system generates sustainable biodegradable alternatives.

Architecture (5-Phase Pipeline):
    Phase 1: Analyze input polymer (properties, structure, applications)
    Phase 2: Load trained surrogates (S_bio, S_mech, S_syn)
    Phase 3: Generate candidates with GFlowNet (biased toward target properties)
    Phase 4: Validate with MD simulation + active learning refinement
    Phase 5: Rank, categorize, and visualize discovered alternatives

Usage:
    engine = PolymerDiscoveryEngine()
    engine.load_models()
    results = engine.discover_alternatives("polyethylene", top_k=20)
    engine.visualize_results(results, output_dir="./results/discovery")
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw, DataStructs
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from models.gflownet import create_gflownet, GFlowNet
from models.surrogate_bio import create_bio_model
from models.surrogate_mech import create_mech_model
from models.surrogate_syn import SynthesizabilityScorer, compute_synthesizability
from simulation.md_simulation import MDSimulator
from evaluation.metrics import compute_all_metrics, compute_diversity
from evaluation.green_ai_metrics import GreenAITracker
from data.preprocessing import smiles_to_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Known polymer database — maps common names to SMILES
# ============================================================
POLYMER_DATABASE = {
    # Commodity plastics (non-biodegradable targets)
    'polyethylene': {
        'smiles': 'CCCCCCCCCCCCCCCC',
        'abbreviation': 'PE',
        'monomer_smiles': 'C=C',
        'category': 'commodity',
        'applications': ['packaging', 'bottles', 'films', 'pipes'],
        'properties': {
            'tensile_strength_mpa': 25.0,
            'tg_celsius': -120.0,
            'flexibility': 0.85,
            'biodegradation_months': 5000,
        },
    },
    'polypropylene': {
        'smiles': 'CC(C)CC(C)CC(C)CC(C)C',
        'abbreviation': 'PP',
        'monomer_smiles': 'CC=C',
        'category': 'commodity',
        'applications': ['packaging', 'automotive', 'textiles', 'containers'],
        'properties': {
            'tensile_strength_mpa': 35.0,
            'tg_celsius': -10.0,
            'flexibility': 0.65,
            'biodegradation_months': 4000,
        },
    },
    'polystyrene': {
        'smiles': 'CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)',
        'abbreviation': 'PS',
        'monomer_smiles': 'C=Cc1ccccc1',
        'category': 'commodity',
        'applications': ['insulation', 'packaging', 'disposable cutlery', 'cups'],
        'properties': {
            'tensile_strength_mpa': 40.0,
            'tg_celsius': 100.0,
            'flexibility': 0.25,
            'biodegradation_months': 5000,
        },
    },
    'pvc': {
        'smiles': 'CC(Cl)CC(Cl)CC(Cl)CC(Cl)',
        'abbreviation': 'PVC',
        'monomer_smiles': 'C=CCl',
        'category': 'commodity',
        'applications': ['pipes', 'flooring', 'window frames', 'cable insulation'],
        'properties': {
            'tensile_strength_mpa': 50.0,
            'tg_celsius': 80.0,
            'flexibility': 0.30,
            'biodegradation_months': 10000,
        },
    },
    'pet': {
        'smiles': 'O=C(c1ccc(C(=O)OCCO)cc1)OCCO',
        'abbreviation': 'PET',
        'monomer_smiles': 'OC(=O)c1ccc(C(=O)O)cc1',
        'category': 'commodity',
        'applications': ['bottles', 'fibers', 'films', 'food containers'],
        'properties': {
            'tensile_strength_mpa': 55.0,
            'tg_celsius': 70.0,
            'flexibility': 0.35,
            'biodegradation_months': 4500,
        },
    },
    'nylon': {
        'smiles': 'O=C(NCCCCCCNC(=O)CCCCC)CCCCC',
        'abbreviation': 'PA6,6',
        'monomer_smiles': 'NCCCCCCN',
        'category': 'engineering',
        'applications': ['textiles', 'automotive', 'gears', 'bearings'],
        'properties': {
            'tensile_strength_mpa': 70.0,
            'tg_celsius': 50.0,
            'flexibility': 0.45,
            'biodegradation_months': 3000,
        },
    },
    'abs': {
        'smiles': 'CC(C#N)CC(c1ccccc1)CC=CC',
        'abbreviation': 'ABS',
        'monomer_smiles': 'C=CC#N',
        'category': 'engineering',
        'applications': ['electronics', 'automotive', 'toys', '3d printing'],
        'properties': {
            'tensile_strength_mpa': 45.0,
            'tg_celsius': 105.0,
            'flexibility': 0.30,
            'biodegradation_months': 5000,
        },
    },
    # Biodegradable polymers (reference benchmarks)
    'pla': {
        'smiles': 'CC(OC(=O)C(C)O)C(=O)O',
        'abbreviation': 'PLA',
        'monomer_smiles': 'CC(O)C(=O)O',
        'category': 'biodegradable',
        'applications': ['packaging', '3d printing', 'medical implants'],
        'properties': {
            'tensile_strength_mpa': 50.0,
            'tg_celsius': 60.0,
            'flexibility': 0.30,
            'biodegradation_months': 6,
        },
    },
    'pcl': {
        'smiles': 'O=C(CCCCCO)OCCCCC',
        'abbreviation': 'PCL',
        'monomer_smiles': 'O=C1CCCCCO1',
        'category': 'biodegradable',
        'applications': ['biomedical', 'packaging', 'adhesives'],
        'properties': {
            'tensile_strength_mpa': 25.0,
            'tg_celsius': -60.0,
            'flexibility': 0.80,
            'biodegradation_months': 24,
        },
    },
    'pga': {
        'smiles': 'OCC(=O)OCC(=O)O',
        'abbreviation': 'PGA',
        'monomer_smiles': 'OCC(=O)O',
        'category': 'biodegradable',
        'applications': ['sutures', 'tissue engineering'],
        'properties': {
            'tensile_strength_mpa': 60.0,
            'tg_celsius': 35.0,
            'flexibility': 0.20,
            'biodegradation_months': 3,
        },
    },
    'pbs': {
        'smiles': 'O=C(CCC(=O)OCCCCO)OCCCCO',
        'abbreviation': 'PBS',
        'monomer_smiles': 'OC(=O)CCC(=O)O',
        'category': 'biodegradable',
        'applications': ['packaging', 'mulch films', 'bags'],
        'properties': {
            'tensile_strength_mpa': 35.0,
            'tg_celsius': -32.0,
            'flexibility': 0.70,
            'biodegradation_months': 4,
        },
    },
    'pha': {
        'smiles': 'CC(CC(=O)OC(C)CC(=O)O)OC(=O)O',
        'abbreviation': 'PHA',
        'monomer_smiles': 'CC(O)CC(=O)O',
        'category': 'biodegradable',
        'applications': ['packaging', 'medical', 'agriculture'],
        'properties': {
            'tensile_strength_mpa': 40.0,
            'tg_celsius': 5.0,
            'flexibility': 0.50,
            'biodegradation_months': 3,
        },
    },
}

# Reverse lookup: abbreviation → name
ABBREVIATION_MAP = {v['abbreviation'].lower(): k for k, v in POLYMER_DATABASE.items()}
# Also map full names with common variations
ABBREVIATION_MAP.update({
    'pe': 'polyethylene', 'pp': 'polypropylene', 'ps': 'polystyrene',
    'polyvinyl chloride': 'pvc', 'polyvinylchloride': 'pvc',
    'polyethylene terephthalate': 'pet', 'nylon 6,6': 'nylon',
    'polylactic acid': 'pla', 'polylactide': 'pla',
    'polycaprolactone': 'pcl', 'polyglycolic acid': 'pga',
    'polybutylene succinate': 'pbs', 'polyhydroxyalkanoate': 'pha',
    'acrylonitrile butadiene styrene': 'abs',
})


@dataclass
class PolymerCandidate:
    """A discovered polymer alternative."""
    rank: int
    smiles: str
    name: str  # Auto-generated descriptive name
    reward: float
    s_bio: float
    s_mech: float
    s_syn: float
    
    # MD simulation results
    is_stable: bool = True
    predicted_tg: float = 0.0
    predicted_tensile: float = 0.0
    predicted_flexibility: float = 0.0
    predicted_biodeg_months: float = 12.0
    
    # Comparison with target
    similarity_to_target: float = 0.0
    mechanical_match_score: float = 0.0
    biodeg_improvement_factor: float = 1.0
    
    # Categorization
    application_category: str = "general"
    strength_category: str = "moderate"  # low / moderate / high
    biodeg_category: str = "moderate"    # fast / moderate / slow
    
    # Polymerizability
    is_polymerizable: bool = False
    polymerizability_score: float = 0.0
    polymerization_mechanisms: List[str] = field(default_factory=list)
    polymerizable_groups: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiscoveryResult:
    """Complete results from a polymer discovery run."""
    target_polymer: str
    target_smiles: str
    target_properties: Dict
    target_applications: List[str]
    
    candidates: List[PolymerCandidate] = field(default_factory=list)
    
    # Summary statistics
    num_generated: int = 0
    num_valid: int = 0
    num_stable: int = 0
    diversity: float = 0.0
    avg_biodeg_improvement: float = 0.0
    
    # Timing & carbon
    total_time_seconds: float = 0.0
    co2_emissions_kg: float = 0.0
    
    # Categories breakdown
    categories: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        return d
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class PolymerDiscoveryEngine:
    """
    Main discovery engine: input a polymer → get sustainable alternatives.
    
    Workflow:
        1. Resolve input (name/abbreviation/SMILES) → target polymer
        2. Analyze target polymer properties
        3. Generate candidates with trained GFlowNet
        4. Score with surrogates (S_bio, S_mech, S_syn)
        5. Validate with MD simulation
        6. Rank by multi-objective score biased toward target's mech properties
        7. Categorize by application suitability
        8. Generate structure images and report
    """
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        device: str = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.checkpoint_dir = checkpoint_dir
        self.gflownet = None
        self.simulator = MDSimulator(noise_level=0.05)
        self.syn_scorer = SynthesizabilityScorer()
        self._models_loaded = False
        
        # Carbon tracking
        self.carbon_tracker = GreenAITracker(
            hardware='cpu' if self.device == 'cpu' else 'mps',
            region='us_average',
        )
    
    def load_models(self, train_if_missing: bool = True):
        """Load trained GFlowNet and surrogate models."""
        gflownet_path = os.path.join(self.checkpoint_dir, 'gflownet_demo.pt')
        bio_path = os.path.join(self.checkpoint_dir, 's_bio_best.pt')
        mech_path = os.path.join(self.checkpoint_dir, 's_mech_best.pt')
        
        has_checkpoints = (
            os.path.exists(gflownet_path) and
            os.path.exists(bio_path) and
            os.path.exists(mech_path)
        )
        
        if not has_checkpoints and train_if_missing:
            logger.info("⚠️  No trained checkpoints found. Running training pipeline first...")
            self._train_pipeline()
        elif not has_checkpoints:
            logger.info("⚠️  No checkpoints found. Creating untrained models (results may be poor).")
        
        # Try to read config from saved checkpoint for architecture reconstruction
        config = {
            'max_atoms': 30,
            'min_atoms': 8,
            'temperature': 1.0,
            'temperature_decay': 0.999,
            'min_temperature': 0.3,
            'epsilon': 0.1,
            'reward_exponent': 2.0,
            'policy_hidden_dim': 256,
            'policy_num_layers': 5,
        }
        
        if os.path.exists(gflownet_path):
            try:
                ckpt = torch.load(gflownet_path, map_location='cpu', weights_only=False)
                if 'config' in ckpt and isinstance(ckpt['config'], dict):
                    saved_cfg = ckpt['config']
                    # Merge saved config (overrides defaults)
                    config.update(saved_cfg)
                    logger.info(f"✅ Loaded architecture config from checkpoint "
                                f"(hidden_dim={config.get('policy_hidden_dim')}, "
                                f"num_layers={config.get('policy_num_layers')})")
                del ckpt  # Free memory
            except Exception as e:
                logger.warning(f"Could not read config from checkpoint: {e}")
        
        # Also check for pipeline_config.json (saved by run_pipeline.py)
        pipeline_config_path = os.path.join(self.checkpoint_dir, 'pipeline_config.json')
        if os.path.exists(pipeline_config_path):
            try:
                import json
                with open(pipeline_config_path) as f:
                    pcfg = json.load(f)
                gfn_cfg = pcfg.get('gflownet', {})
                for key in ['policy_hidden_dim', 'policy_num_layers', 'policy_dropout',
                            'max_atoms', 'min_atoms', 'reward_exponent',
                            'alpha_bio', 'alpha_mech', 'alpha_syn',
                            'ester_bonus', 'amide_bonus', 'hydroxyl_bonus',
                            'halogen_penalty', 'max_shaping_bonus',
                            'size_bonus_threshold', 'size_bonus']:
                    if key in gfn_cfg:
                        config[key] = gfn_cfg[key]
                logger.info(f"✅ Merged config from pipeline_config.json")
            except Exception as e:
                logger.warning(f"Could not read pipeline_config.json: {e}")
        
        self.gflownet = create_gflownet(config, device=self.device)
        
        # Load checkpoints
        if os.path.exists(gflownet_path):
            self.gflownet.load(gflownet_path)
            logger.info(f"✅ GFlowNet loaded from {gflownet_path}")
        
        if os.path.exists(bio_path):
            self.gflownet.reward_fn.bio_model.load_state_dict(
                torch.load(bio_path, map_location=self.device, weights_only=True)
            )
            logger.info(f"✅ S_bio loaded from {bio_path}")
        
        if os.path.exists(mech_path):
            self.gflownet.reward_fn.mech_model.load_state_dict(
                torch.load(mech_path, map_location=self.device, weights_only=True)
            )
            logger.info(f"✅ S_mech loaded from {mech_path}")
        
        self._models_loaded = True
        logger.info(f"🔧 Device: {self.device}")
    
    def _train_pipeline(self):
        """Run the training pipeline if checkpoints don't exist."""
        from demo.demo_pipeline import run_demo_pipeline
        run_demo_pipeline()
    
    def resolve_polymer(self, query: str) -> Dict:
        """
        Resolve a polymer query (name, abbreviation, or SMILES) to
        its full database entry.
        
        Args:
            query: e.g. "polyethylene", "PE", "C=C", "CCCC"
        
        Returns:
            Polymer info dictionary
        """
        query_lower = query.lower().strip()
        
        # 1. Try exact name match
        if query_lower in POLYMER_DATABASE:
            info = POLYMER_DATABASE[query_lower].copy()
            info['resolved_name'] = query_lower
            return info
        
        # 2. Try abbreviation
        if query_lower in ABBREVIATION_MAP:
            name = ABBREVIATION_MAP[query_lower]
            info = POLYMER_DATABASE[name].copy()
            info['resolved_name'] = name
            return info
        
        # 3. Try partial match
        for name, info in POLYMER_DATABASE.items():
            if query_lower in name or name in query_lower:
                result = info.copy()
                result['resolved_name'] = name
                return result
        
        # 4. Try as SMILES
        mol = Chem.MolFromSmiles(query)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            # Try to match against known SMILES
            for name, info in POLYMER_DATABASE.items():
                known_mol = Chem.MolFromSmiles(info['smiles'])
                if known_mol is not None:
                    known_smiles = Chem.MolToSmiles(known_mol)
                    if smiles == known_smiles:
                        result = info.copy()
                        result['resolved_name'] = name
                        return result
            
            # Unknown SMILES — create entry from structure
            return self._analyze_unknown_polymer(query, mol)
        
        raise ValueError(
            f"Could not resolve polymer '{query}'. "
            f"Known polymers: {', '.join(POLYMER_DATABASE.keys())}. "
            f"You can also provide a valid SMILES string."
        )
    
    def _analyze_unknown_polymer(self, smiles: str, mol) -> Dict:
        """Analyze an unknown polymer from its SMILES."""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Estimate properties using MD simulator
        sim_result = self.simulator.simulate(smiles)
        
        # Guess flexibility
        flexibility = min(num_rot / max(mol.GetNumHeavyAtoms(), 1) * 2.0, 0.9)
        
        return {
            'smiles': Chem.MolToSmiles(mol),
            'abbreviation': 'CUSTOM',
            'monomer_smiles': smiles,
            'category': 'custom',
            'resolved_name': f'custom_polymer ({smiles})',
            'applications': ['general'],
            'properties': {
                'tensile_strength_mpa': sim_result.predicted_tensile,
                'tg_celsius': sim_result.predicted_tg,
                'flexibility': sim_result.predicted_flexibility,
                'biodegradation_months': sim_result.predicted_biodeg_rate,
            },
        }
    
    def discover_alternatives(
        self,
        target_polymer: str,
        num_candidates: int = 500,
        top_k: int = 20,
        min_biodeg_improvement: float = 1.5,
        similarity_weight: float = 0.3,
    ) -> DiscoveryResult:
        """
        Discover biodegradable alternatives to a target polymer.
        
        Args:
            target_polymer: Name, abbreviation, or SMILES of target polymer
            num_candidates: Number of candidates to generate
            top_k: Number of top alternatives to return
            min_biodeg_improvement: Minimum biodegradation improvement factor
            similarity_weight: How much to weight mechanical property similarity
        
        Returns:
            DiscoveryResult with ranked alternatives
        """
        if not self._models_loaded:
            self.load_models()
        
        self.carbon_tracker.start()
        start_time = time.time()
        
        # ─── Phase 1: Resolve & Analyze Target ───
        print("\n" + "═" * 70)
        print("  🔬 GFlowNet Polymer Discovery Engine")
        print("═" * 70)
        
        target_info = self.resolve_polymer(target_polymer)
        target_name = target_info['resolved_name']
        target_smiles = target_info['smiles']
        target_props = target_info['properties']
        target_apps = target_info.get('applications', [])
        
        print(f"\n  📋 Target Polymer: {target_name.upper()} ({target_info['abbreviation']})")
        print(f"     SMILES: {target_smiles}")
        print(f"     Applications: {', '.join(target_apps)}")
        print(f"     Properties:")
        print(f"       Tensile Strength: {target_props['tensile_strength_mpa']:.1f} MPa")
        print(f"       Glass Transition: {target_props['tg_celsius']:.1f} °C")
        print(f"       Flexibility:      {target_props['flexibility']:.2f}")
        print(f"       Biodeg Time:      {target_props['biodegradation_months']:.0f} months")
        print(f"\n  ⚠️  This polymer takes ~{target_props['biodegradation_months']/12:.0f} YEARS to degrade!")
        print(f"  🎯 Goal: Find alternatives that biodegrade in <24 months")
        
        # Get target fingerprint for similarity
        target_mol = Chem.MolFromSmiles(target_smiles)
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048) if target_mol else None
        
        # ─── Phase 2: Generate Candidates ───
        print(f"\n\n  🧪 Phase 2: Generating {num_candidates} candidate molecules...")
        print("  " + "─" * 50)
        
        generated = self.gflownet.generate_molecules(
            num_molecules=num_candidates,
            unique=True,
        )
        
        valid_results = [r for r in generated if r.get('valid', False)]
        print(f"  ✅ Generated: {len(generated)}, Valid: {len(valid_results)}")
        
        # ─── Phase 3: Score & Filter ───
        print(f"\n  📊 Phase 3: Scoring and filtering candidates...")
        print("  " + "─" * 50)
        
        scored_candidates = []
        n_chem_filtered = 0
        for r in valid_results:
            smiles = r['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # ── Chemistry hard filter ──────────────────────────
            # Reject molecules with exotic bond types that are
            # unrealistic for lab-synthesizable biodegradable polymers.
            _no_pat = Chem.MolFromSmarts('[N]-[O]')
            _nn_pat = Chem.MolFromSmarts('[N]-[N]')
            _oo_pat = Chem.MolFromSmarts('[O]-[O]')
            _neo_pat = Chem.MolFromSmarts('[N]=[O]')
            has_bad_bond = False
            for pat in [_no_pat, _nn_pat, _oo_pat, _neo_pat]:
                if pat and mol.HasSubstructMatch(pat):
                    has_bad_bond = True
                    break
            # Also reject exotic atoms (should already be blocked but safety net)
            bad_atoms = any(a.GetSymbol() not in ('C', 'N', 'O') for a in mol.GetAtoms())
            if has_bad_bond or bad_atoms:
                n_chem_filtered += 1
                continue
            
            # Compute similarity to target
            similarity = 0.0
            if target_fp is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(target_fp, fp)
            
            # Combined score:
            # - High reward (bio + mech + syn)
            # - Bonus for mechanical similarity to target
            # - Must have high biodegradability
            mech_match = self._compute_mechanical_match(r, target_props)
            
            discovery_score = (
                r['reward'] * (1.0 - similarity_weight) +
                mech_match * similarity_weight
            )
            
            scored_candidates.append({
                **r,
                'similarity': similarity,
                'mech_match': mech_match,
                'discovery_score': discovery_score,
            })
        
        print(f"  🧹 Chemistry filter: removed {n_chem_filtered} molecules with N-O/N-N/O-O bonds")
        print(f"  ✅ Passed filter: {len(scored_candidates)} clean candidates")
        
        # Sort by discovery score
        scored_candidates.sort(key=lambda x: x['discovery_score'], reverse=True)
        
        # ─── Phase 4: MD Simulation Validation ───
        top_for_sim = scored_candidates[:min(top_k * 3, len(scored_candidates))]
        
        print(f"\n  🔬 Phase 4: MD Simulation on top {len(top_for_sim)} candidates...")
        print("  " + "─" * 50)
        
        sim_smiles = [c['smiles'] for c in top_for_sim]
        sim_results = self.simulator.simulate_batch(sim_smiles)
        
        # Merge simulation results
        validated = []
        for candidate, sim in zip(top_for_sim, sim_results):
            # Biodeg improvement factor
            target_biodeg = target_props['biodegradation_months']
            biodeg_improvement = target_biodeg / max(sim.predicted_biodeg_rate, 0.5)
            
            if sim.is_stable and biodeg_improvement >= min_biodeg_improvement:
                validated.append({
                    **candidate,
                    'sim': sim,
                    'biodeg_improvement': biodeg_improvement,
                })
        
        stable_count = sum(1 for s in sim_results if s.is_stable)
        print(f"  ✅ Stable: {stable_count}/{len(sim_results)}")
        print(f"  ✅ Meet biodeg threshold: {len(validated)}")
        
        # ─── Phase 5: Rank & Categorize ───
        print(f"\n  🏆 Phase 5: Ranking and categorizing top alternatives...")
        print("  " + "─" * 50)
        
        # Sort validated by combined discovery score
        validated.sort(key=lambda x: x['discovery_score'] * x['biodeg_improvement'], reverse=True)
        
        # Build PolymerCandidate objects
        candidates = []
        for rank, v in enumerate(validated[:top_k], 1):
            sim = v['sim']
            
            # Auto-generate name
            name = self._generate_polymer_name(v['smiles'], rank)
            
            # Categorize
            app_category = self._categorize_application(sim, target_apps)
            strength_cat = 'high' if sim.predicted_tensile > 40 else ('moderate' if sim.predicted_tensile > 20 else 'low')
            biodeg_cat = 'fast' if sim.predicted_biodeg_rate < 12 else ('moderate' if sim.predicted_biodeg_rate < 36 else 'slow')
            
            # Compute polymerizability
            try:
                from data.real_polymer_data import compute_polymerizability
                poly_info = compute_polymerizability(v['smiles'])
            except ImportError:
                poly_info = {'is_polymerizable': False, 'score': 0.0, 'mechanisms': [], 'groups_found': []}
            
            candidate = PolymerCandidate(
                rank=rank,
                smiles=v['smiles'],
                name=name,
                reward=v['reward'],
                s_bio=v['s_bio'],
                s_mech=v['s_mech'],
                s_syn=v['s_syn'],
                is_stable=sim.is_stable,
                predicted_tg=sim.predicted_tg,
                predicted_tensile=sim.predicted_tensile,
                predicted_flexibility=sim.predicted_flexibility,
                predicted_biodeg_months=sim.predicted_biodeg_rate,
                similarity_to_target=v['similarity'],
                mechanical_match_score=v['mech_match'],
                biodeg_improvement_factor=v['biodeg_improvement'],
                application_category=app_category,
                strength_category=strength_cat,
                biodeg_category=biodeg_cat,
                is_polymerizable=poly_info['is_polymerizable'],
                polymerizability_score=poly_info['score'],
                polymerization_mechanisms=poly_info['mechanisms'],
                polymerizable_groups=poly_info['groups_found'],
            )
            candidates.append(candidate)
        
        # Compute diversity of discovered set
        discovered_smiles = [c.smiles for c in candidates]
        diversity = compute_diversity(discovered_smiles) if len(discovered_smiles) >= 2 else 0.0
        
        total_time = time.time() - start_time
        emissions = self.carbon_tracker.stop()
        
        # Build result
        result = DiscoveryResult(
            target_polymer=target_name,
            target_smiles=target_smiles,
            target_properties=target_props,
            target_applications=target_apps,
            candidates=candidates,
            num_generated=len(generated),
            num_valid=len(valid_results),
            num_stable=stable_count,
            diversity=diversity,
            avg_biodeg_improvement=np.mean([c.biodeg_improvement_factor for c in candidates]) if candidates else 0.0,
            total_time_seconds=total_time,
            co2_emissions_kg=emissions,
            categories=self._build_category_breakdown(candidates),
        )
        
        # ─── Print Results ───
        self._print_results(result)
        
        return result
    
    def _compute_mechanical_match(self, result: Dict, target_props: Dict) -> float:
        """Compute how well a candidate matches the target's mechanical properties."""
        # Normalize target properties to [0,1] scale
        target_tensile = target_props['tensile_strength_mpa'] / 100.0
        target_flex = target_props['flexibility']
        
        # Get candidate mech details
        mech = result.get('mech_details', {})
        cand_tensile = mech.get('tensile', result.get('s_mech', 0.5))
        cand_flex = mech.get('flexibility', 0.5)
        
        # Similarity in property space
        tensile_sim = 1.0 - abs(target_tensile - cand_tensile)
        flex_sim = 1.0 - abs(target_flex - cand_flex)
        
        return (tensile_sim + flex_sim) / 2.0
    
    def _generate_polymer_name(self, smiles: str, rank: int) -> str:
        """Generate a descriptive name for a polymer candidate."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"BioPolymer-{rank:03d}"
        
        # Analyze functional groups
        ester = Chem.MolFromSmarts('[#6](=[#8])-[#8]')
        amide = Chem.MolFromSmarts('[#6](=[#8])-[#7]')
        hydroxyl = Chem.MolFromSmarts('[OX2H]')
        
        has_ester = ester and mol.HasSubstructMatch(ester)
        has_amide = amide and mol.HasSubstructMatch(amide)
        has_hydroxyl = hydroxyl and mol.HasSubstructMatch(hydroxyl)
        
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_atoms = mol.GetNumHeavyAtoms()
        
        prefix = "Bio"
        if has_ester and has_amide:
            base = "EsterAmide"
        elif has_ester:
            base = "Ester"
        elif has_amide:
            base = "Amide"
        elif has_hydroxyl:
            base = "Hydroxy"
        elif num_rings > 0:
            base = "Cyclic"
        else:
            base = "Linear"
        
        size = "Oligo" if num_atoms < 8 else "Poly"
        
        return f"{prefix}{base}-{size}-{rank:03d}"
    
    def _categorize_application(self, sim, target_apps: List[str]) -> str:
        """Categorize by best-fit application based on properties."""
        if sim.predicted_tensile > 40 and sim.predicted_flexibility < 0.4:
            return "structural (rigid packaging, containers)"
        elif sim.predicted_biodeg_rate < 12:
            return "fast-degrading (single-use, agricultural)"
        elif sim.predicted_flexibility > 0.6:
            return "flexible (films, bags, wraps)"
        elif sim.predicted_tensile > 30:
            return "general purpose (mixed applications)"
        elif sim.predicted_biodeg_rate < 24:
            return "compostable (food service, packaging)"
        else:
            return "specialty (coatings, adhesives)"
    
    def _build_category_breakdown(self, candidates: List[PolymerCandidate]) -> Dict:
        """Build category statistics."""
        cats = {}
        for c in candidates:
            cat = c.application_category
            if cat not in cats:
                cats[cat] = {'count': 0, 'avg_reward': 0, 'avg_biodeg': 0, 'smiles': []}
            cats[cat]['count'] += 1
            cats[cat]['avg_reward'] += c.reward
            cats[cat]['avg_biodeg'] += c.predicted_biodeg_months
            cats[cat]['smiles'].append(c.smiles)
        
        for cat in cats:
            n = cats[cat]['count']
            cats[cat]['avg_reward'] /= n
            cats[cat]['avg_biodeg'] /= n
        
        return cats
    
    def _print_results(self, result: DiscoveryResult):
        """Print discovery results in a beautiful format."""
        print("\n\n" + "═" * 70)
        print(f"  🎉 DISCOVERY RESULTS: Alternatives to {result.target_polymer.upper()}")
        print("═" * 70)
        
        print(f"\n  Pipeline Statistics:")
        print(f"    Generated:     {result.num_generated} candidates")
        print(f"    Valid:         {result.num_valid}")
        print(f"    MD Validated:  {result.num_stable}")
        print(f"    Final Top-K:   {len(result.candidates)}")
        print(f"    Diversity:     {result.diversity:.4f}")
        print(f"    Avg Biodeg Improvement: {result.avg_biodeg_improvement:.1f}×")
        print(f"    Time:          {result.total_time_seconds:.1f}s")
        print(f"    CO₂:           {result.co2_emissions_kg:.6f} kg")
        
        if result.candidates:
            target_biodeg = result.target_properties['biodegradation_months']
            
            print(f"\n  {'Rank':<5} {'Name':<25} {'SMILES':<30} {'R(x)':<7} {'Biodeg':<8} {'Improv':<8} {'Category'}")
            print("  " + "─" * 110)
            
            for c in result.candidates:
                biodeg_str = f"{c.predicted_biodeg_months:.1f}mo"
                improv_str = f"{c.biodeg_improvement_factor:.0f}×"
                print(
                    f"  {c.rank:<5} {c.name:<25} "
                    f"{c.smiles[:28]:<30} "
                    f"{c.reward:<7.4f} "
                    f"{biodeg_str:<8} "
                    f"{improv_str:<8} "
                    f"{c.application_category}"
                )
            
            print("\n  📊 Category Breakdown:")
            for cat, info in result.categories.items():
                print(f"    • {cat}: {info['count']} candidates (avg biodeg: {info['avg_biodeg']:.1f} months)")
        else:
            print("\n  ⚠️  No candidates met the biodegradation improvement threshold.")
            print("      Try lowering min_biodeg_improvement or generating more candidates.")
        
        print("\n" + "═" * 70)
    
    def visualize_results(
        self,
        result: DiscoveryResult,
        output_dir: str = './results/discovery',
    ):
        """
        Generate comprehensive visualization of discovery results.
        
        Creates:
            1. Structure grid of top candidates (2D molecular structures)
            2. Property comparison radar/bar chart
            3. Biodegradation improvement chart
            4. Target vs alternatives comparison
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        os.makedirs(output_dir, exist_ok=True)
        
        candidates = result.candidates
        if not candidates:
            logger.warning("No candidates to visualize")
            return
        
        # ─── 1. Molecule Structure Grid ───
        self._plot_structure_grid(candidates, result, output_dir)
        
        # ─── 2. Property Comparison ───
        self._plot_property_comparison(candidates, result, output_dir)
        
        # ─── 3. Biodegradation Improvement ───
        self._plot_biodeg_improvement(candidates, result, output_dir)
        
        # ─── 4. Score Components ───
        self._plot_score_breakdown(candidates, result, output_dir)
        
        # ─── 5. Summary Dashboard ───
        self._plot_dashboard(candidates, result, output_dir)
        
        # Save results JSON
        result.save(os.path.join(output_dir, 'discovery_results.json'))
        
        print(f"\n  📁 All visualizations saved to {output_dir}/")
        print(f"     • structures_grid.png     — 2D molecular structures")
        print(f"     • property_comparison.png  — Mechanical properties comparison")
        print(f"     • biodeg_improvement.png   — Biodegradation improvement chart")
        print(f"     • score_breakdown.png      — Reward component breakdown")
        print(f"     • discovery_dashboard.png  — Summary dashboard")
        print(f"     • individual_structures/   — Per-candidate structure images")
        print(f"     • discovery_results.json   — Machine-readable results")
    
    def _plot_structure_grid(self, candidates, result, output_dir):
        """Plot 2D structures of all discovered polymers."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        mols = []
        legends = []
        
        for c in candidates:
            mol = Chem.MolFromSmiles(c.smiles)
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
                legends.append(
                    f"#{c.rank} {c.name}\n"
                    f"Biodeg: {c.predicted_biodeg_months:.1f} mo "
                    f"({c.biodeg_improvement_factor:.0f}× better)\n"
                    f"R={c.reward:.3f}"
                )
        
        if not mols:
            return
        
        # Grid image
        n_per_row = min(5, len(mols))
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=n_per_row,
            subImgSize=(450, 350),
            legends=legends,
        )
        img.save(os.path.join(output_dir, 'structures_grid.png'))
        logger.info(f"✅ Structure grid saved ({len(mols)} molecules)")
        
        # Individual structure images
        ind_dir = os.path.join(output_dir, 'individual_structures')
        os.makedirs(ind_dir, exist_ok=True)
        
        for i, (mol, c) in enumerate(zip(mols, candidates)):
            img_single = Draw.MolToImage(mol, size=(500, 400))
            img_single.save(os.path.join(ind_dir, f'candidate_{c.rank:03d}_{c.name}.png'))
        
        # Also plot target polymer structure
        target_mol = Chem.MolFromSmiles(result.target_smiles)
        if target_mol is not None:
            AllChem.Compute2DCoords(target_mol)
            target_img = Draw.MolToImage(target_mol, size=(500, 400))
            target_img.save(os.path.join(ind_dir, f'TARGET_{result.target_polymer}.png'))
    
    def _plot_property_comparison(self, candidates, result, output_dir):
        """Bar chart comparing properties of alternatives vs target."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        target_props = result.target_properties
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Tensile Strength
        ax = axes[0]
        names = ['Target\n(' + result.target_polymer.upper() + ')'] + [f'#{c.rank}' for c in candidates[:10]]
        tensiles = [target_props['tensile_strength_mpa']] + [c.predicted_tensile for c in candidates[:10]]
        colors = ['#FF5252'] + ['#4CAF50'] * len(candidates[:10])
        bars = ax.barh(names, tensiles, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xlabel('Tensile Strength (MPa)')
        ax.set_title('Tensile Strength Comparison')
        ax.invert_yaxis()
        for bar, val in zip(bars, tensiles):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                   va='center', fontsize=9)
        
        # Flexibility
        ax = axes[1]
        flexes = [target_props['flexibility']] + [c.predicted_flexibility for c in candidates[:10]]
        bars = ax.barh(names, flexes, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xlabel('Flexibility Score')
        ax.set_title('Flexibility Comparison')
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()
        
        # Biodegradation Time
        ax = axes[2]
        biodeg = [target_props['biodegradation_months']] + [c.predicted_biodeg_months for c in candidates[:10]]
        # Log scale for better viz
        colors_biodeg = ['#FF5252'] + ['#4CAF50'] * len(candidates[:10])
        bars = ax.barh(names, biodeg, color=colors_biodeg, alpha=0.85, edgecolor='white')
        ax.set_xlabel('Biodegradation Time (months, log scale)')
        ax.set_xscale('log')
        ax.set_title('Biodegradation Time (lower = better)')
        ax.invert_yaxis()
        
        plt.suptitle(
            f'Property Comparison: Alternatives vs {result.target_polymer.upper()}',
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'property_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_biodeg_improvement(self, candidates, result, output_dir):
        """Chart showing biodegradation improvement factors."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = [f'#{c.rank} {c.name}' for c in candidates[:15]]
        improvements = [c.biodeg_improvement_factor for c in candidates[:15]]
        
        colors = ['#2E7D32' if imp > 100 else '#4CAF50' if imp > 50 else '#81C784' for imp in improvements]
        
        bars = ax.bar(range(len(names)), improvements, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Biodegradation Improvement Factor (×)')
        ax.set_title(
            f'How Much Faster Do Alternatives Biodegrade vs {result.target_polymer.upper()}?\n'
            f'(Target degrades in ~{result.target_properties["biodegradation_months"]/12:.0f} years)',
            fontsize=14
        )
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{val:.0f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Same as target')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'biodeg_improvement.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_breakdown(self, candidates, result, output_dir):
        """Stacked bar chart of reward components."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        names = [f'#{c.rank}' for c in candidates[:15]]
        s_bio = [c.s_bio for c in candidates[:15]]
        s_mech = [c.s_mech for c in candidates[:15]]
        s_syn = [c.s_syn for c in candidates[:15]]
        
        x = np.arange(len(names))
        width = 0.25
        
        ax.bar(x - width, s_bio, width, label='S_bio (Biodegradability)', color='#4CAF50', alpha=0.85)
        ax.bar(x, s_mech, width, label='S_mech (Mechanical)', color='#2196F3', alpha=0.85)
        ax.bar(x + width, s_syn, width, label='S_syn (Synthesizability)', color='#FF9800', alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Score')
        ax.set_title('Reward Component Breakdown for Top Candidates')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_breakdown.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dashboard(self, candidates, result, output_dir):
        """Create a comprehensive summary dashboard."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # --- Title ---
        fig.suptitle(
            f'GFlowNet Polymer Discovery Dashboard\n'
            f'Sustainable Alternatives to {result.target_polymer.upper()} ({result.target_properties["biodegradation_months"]/12:.0f}-year degradation)',
            fontsize=18, fontweight='bold', y=0.98
        )
        
        # --- 1. Top candidate structures (first 4) ---
        ax_structs = fig.add_subplot(gs[0, :2])
        mols = []
        for c in candidates[:8]:
            mol = Chem.MolFromSmiles(c.smiles)
            if mol:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
        
        if mols:
            img = Draw.MolsToGridImage(
                mols[:8],
                molsPerRow=4,
                subImgSize=(300, 220),
                legends=[f'#{c.rank} {c.name}' for c in candidates[:8]],
            )
            ax_structs.imshow(img)
        ax_structs.axis('off')
        ax_structs.set_title('Top 8 Discovered Structures', fontsize=14)
        
        # --- 2. Summary stats ---
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')
        stats_text = (
            f"Pipeline Summary\n"
            f"{'─'*30}\n"
            f"Generated:     {result.num_generated}\n"
            f"Valid:         {result.num_valid}\n"
            f"MD Stable:     {result.num_stable}\n"
            f"Final Top-K:   {len(result.candidates)}\n"
            f"Diversity:     {result.diversity:.4f}\n"
            f"{'─'*30}\n"
            f"Avg Biodeg:    {result.avg_biodeg_improvement:.1f}×\n"
            f"Time:          {result.total_time_seconds:.1f}s\n"
            f"CO₂:           {result.co2_emissions_kg:.6f} kg\n"
        )
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=12, verticalalignment='center', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
        
        # --- 3. Biodeg improvement bars ---
        ax_biodeg = fig.add_subplot(gs[1, :])
        names = [f'#{c.rank} {c.name[:15]}' for c in candidates[:12]]
        improvements = [c.biodeg_improvement_factor for c in candidates[:12]]
        colors = ['#2E7D32' if imp > 100 else '#4CAF50' if imp > 50 else '#81C784' for imp in improvements]
        bars = ax_biodeg.bar(range(len(names)), improvements, color=colors, alpha=0.85, edgecolor='white')
        ax_biodeg.set_xticks(range(len(names)))
        ax_biodeg.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax_biodeg.set_ylabel('Biodeg Improvement (×)')
        ax_biodeg.set_title('Biodegradation Improvement Factor vs Target', fontsize=14)
        ax_biodeg.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, improvements):
            ax_biodeg.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                          f'{val:.0f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # --- 4. Score components ---
        ax_scores = fig.add_subplot(gs[2, 0])
        s_bio_vals = [c.s_bio for c in candidates[:10]]
        s_mech_vals = [c.s_mech for c in candidates[:10]]
        s_syn_vals = [c.s_syn for c in candidates[:10]]
        x = range(len(s_bio_vals))
        ax_scores.plot(x, s_bio_vals, 'o-', label='S_bio', color='#4CAF50')
        ax_scores.plot(x, s_mech_vals, 's-', label='S_mech', color='#2196F3')
        ax_scores.plot(x, s_syn_vals, '^-', label='S_syn', color='#FF9800')
        ax_scores.set_xlabel('Candidate Rank')
        ax_scores.set_ylabel('Score')
        ax_scores.set_title('Surrogate Scores')
        ax_scores.legend(fontsize=9)
        ax_scores.grid(True, alpha=0.3)
        
        # --- 5. Category pie chart ---
        ax_cats = fig.add_subplot(gs[2, 1])
        cat_names = list(result.categories.keys())
        cat_counts = [result.categories[c]['count'] for c in cat_names]
        if cat_counts:
            short_names = [n.split('(')[0].strip() for n in cat_names]
            ax_cats.pie(cat_counts, labels=short_names, autopct='%1.0f%%',
                       colors=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#FF5252'])
        ax_cats.set_title('Application Categories')
        
        # --- 6. Tensile vs Biodeg scatter ---
        ax_scatter = fig.add_subplot(gs[2, 2])
        tensiles = [c.predicted_tensile for c in candidates]
        biodeg = [c.predicted_biodeg_months for c in candidates]
        sc = ax_scatter.scatter(tensiles, biodeg, c=[c.reward for c in candidates],
                               cmap='YlOrRd', s=80, alpha=0.8, edgecolors='white')
        ax_scatter.set_xlabel('Tensile Strength (MPa)')
        ax_scatter.set_ylabel('Biodeg Time (months)')
        ax_scatter.set_title('Strength vs Biodegradability')
        plt.colorbar(sc, ax=ax_scatter, label='Reward')
        
        # Target marker
        ax_scatter.scatter(
            [result.target_properties['tensile_strength_mpa']],
            [min(result.target_properties['biodegradation_months'], max(biodeg) * 1.2 if biodeg else 100)],
            marker='X', s=200, c='red', zorder=5, label='Target'
        )
        ax_scatter.legend(fontsize=9)
        ax_scatter.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'discovery_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Dashboard saved")


def main():
    """
    Main entry point for polymer discovery.
    Run: python -m discovery.polymer_discovery
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='GFlowNet Polymer Discovery Engine')
    parser.add_argument('polymer', nargs='?', default='polyethylene',
                       help='Target polymer name, abbreviation, or SMILES (default: polyethylene)')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top alternatives to return')
    parser.add_argument('--num-candidates', type=int, default=500,
                       help='Number of candidates to generate')
    parser.add_argument('--output-dir', type=str, default='./results/discovery',
                       help='Output directory for results')
    parser.add_argument('--min-improvement', type=float, default=1.5,
                       help='Minimum biodegradation improvement factor')
    
    args = parser.parse_args()
    
    engine = PolymerDiscoveryEngine()
    engine.load_models()
    
    result = engine.discover_alternatives(
        target_polymer=args.polymer,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        min_biodeg_improvement=args.min_improvement,
    )
    
    engine.visualize_results(result, output_dir=args.output_dir)
    
    return result


if __name__ == "__main__":
    main()
