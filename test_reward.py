"""Quick test of updated reward function with known molecules."""
import torch, os
from configs.pipeline_config import PipelineConfig
from models.surrogate_bio import BiodegradabilityPredictor
from models.surrogate_mech import MechanicalPropertiesPredictor
from models.surrogate_syn import SynthesizabilityScorer
from models.gflownet import RewardFunction

config = PipelineConfig()
device = config.resolve_device()
print(f"Device: {device}, min_atoms: {config.gflownet.min_atoms}")

bio = BiodegradabilityPredictor(
    config.data.atom_feature_dim, config.data.bond_feature_dim,
    config.surrogate.hidden_dim, config.surrogate.num_layers,
)
mech = MechanicalPropertiesPredictor(
    config.data.atom_feature_dim, config.data.bond_feature_dim,
    config.surrogate.hidden_dim, config.surrogate.num_layers,
)
syn = SynthesizabilityScorer()

bio_ckpt = os.path.join(config.checkpoint_dir, "surrogate_bio.pt")
mech_ckpt = os.path.join(config.checkpoint_dir, "surrogate_mech.pt")
if os.path.exists(bio_ckpt):
    bio.load_state_dict(torch.load(bio_ckpt, map_location=device, weights_only=True))
    print("Loaded trained bio model")
if os.path.exists(mech_ckpt):
    mech.load_state_dict(torch.load(mech_ckpt, map_location=device, weights_only=True))
    print("Loaded trained mech model")

gf = config.gflownet
rf = RewardFunction(
    bio, mech, syn,
    alpha_bio=gf.alpha_bio, alpha_mech=gf.alpha_mech, alpha_syn=gf.alpha_syn,
    reward_exponent=gf.reward_exponent, reward_min=gf.reward_min, device=device,
    ester_bonus=gf.ester_bonus, amide_bonus=gf.amide_bonus,
    hydroxyl_bonus=gf.hydroxyl_bonus, size_bonus_threshold=gf.size_bonus_threshold,
    size_bonus=gf.size_bonus, halogen_penalty=gf.halogen_penalty,
    max_shaping_bonus=gf.max_shaping_bonus,
)

tests = [
    ("CC(O)C(=O)O",          "Lactic acid (PLA)"),
    ("O=C1CCCCCO1",          "Caprolactone (PCL)"),
    ("O=C1COC(=O)CO1",       "Glycolide (PGA)"),
    ("COC(=O)CCCC(=O)OC",    "DiMe adipate"),
    ("OCC(O)CO",              "Glycerol"),
    ("OC(=O)CCCC(=O)O",      "Adipic acid"),
    ("OC(=O)C(O)C(O)C(=O)O", "Tartaric acid"),
    ("CCCCCCCCCC",            "Decane (PE)"),
    ("O=COS",                 "O=COS (S!)"),
    ("CNN(C)C(=O)OO",         "Peroxide"),
    ("O=C(N=S)N=S",           "N=S bonds"),
    ("C",                      "Methane (tiny)"),
    ("CC",                     "Ethane (tiny)"),
]

header = f"{'Molecule':<25} {'R':>8} {'bio':>6} {'mech':>6} {'syn':>6} {'shape':>7} {'#at':>4}"
print("\n" + header)
print("-" * len(header))
for smi, name in tests:
    r = rf.compute_reward(smi)
    na = r.get("num_atoms", 0)
    print(f"{name:<25} {r['reward']:8.4f} {r['s_bio']:6.3f} {r['s_mech']:6.3f} {r['s_syn']:6.3f} {r['shaping_bonus']:7.3f} {na:4d}")

# Test MD simulation calibration
print("\n\n=== MD Simulation Calibration ===")
from simulation.md_simulation import MDSimulator
sim = MDSimulator(noise_level=0.0)

md_tests = [
    ("CC(O)C(=O)O",          "Lactic acid"),
    ("O=C1CCCCCO1",          "Caprolactone"),
    ("O=C1COC(=O)CO1",       "Glycolide"),
    ("COC(=O)CCCC(=O)OC",    "DiMe adipate"),
    ("CCCCCCCCCC",            "Decane (PE)"),
    ("O=COS",                 "O=COS (S)"),
]

print(f"{'Molecule':<25} {'Biodeg (mo)':>12} {'Tensile (MPa)':>14} {'Tg (C)':>8} {'Stable':>7}")
print("-" * 70)
for smi, name in md_tests:
    r = sim.simulate(smi)
    print(f"{name:<25} {r.predicted_biodeg_rate:12.1f} {r.predicted_tensile:14.1f} {r.predicted_tg:8.1f} {'Y' if r.is_stable else 'N':>7}")
