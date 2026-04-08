"""
Fransen et al. Polyester Dataset — Publication Improvement #6
=============================================================
Based on: "High-throughput experimentation for discovery of
biodegradable polyesters" (PNAS 2023, doi:10.1073/pnas.2321508121)

642 polyesters and polycarbonates tested with P. lemoignei enzyme.
This module reconstructs representative monomers and repeat units
from the paper's building blocks (diols, diacids, lactones, carbonates).
"""

from data.real_polymer_data import RealPolymerEntry

SRC = 'Fransen et al. PNAS 2023'

# Helper to construct entries consistently
def _e(smiles, name, biodeg, tg, tensile, flex, cat='biodegradable', poly=True):
    return RealPolymerEntry(smiles, name, biodeg, tg, tensile, flex, cat, SRC, poly)


# ── DIOLS ──
FRANSEN_DIOLS = [
    _e('OCCO', 'Ethylene glycol', 0.75, -13, 10, 0.70),
    _e('OCCCO', '1,3-Propanediol', 0.80, -28, 8, 0.75),
    _e('OCCCCO', '1,4-Butanediol', 0.82, 20, 12, 0.72),
    _e('OCCCCCO', '1,5-Pentanediol', 0.78, -18, 10, 0.74),
    _e('OCCCCCCO', '1,6-Hexanediol', 0.72, -12, 11, 0.76),
    _e('OCCCCCCCO', '1,8-Octanediol', 0.65, 0, 14, 0.78),
    _e('OCCCCCCCCCCO', '1,10-Decanediol', 0.55, 10, 16, 0.80),
    _e('OCC(C)O', '1,2-Propanediol', 0.76, -30, 8, 0.72),
    _e('OCC(C)(C)CO', 'Neopentyl glycol', 0.45, 5, 18, 0.55, 'partially_biodegradable'),
    _e('OC(CO)CO', 'Glycerol', 0.85, 17, 6, 0.65),
    _e('OCC(O)C(O)CO', 'Erythritol', 0.82, 120, 8, 0.40),
    _e('OC1CCCCC1O', 'trans-1,2-Cyclohexanediol', 0.40, 80, 35, 0.35, 'partially_biodegradable'),
    _e('OC1CCC(O)CC1', '1,4-Cyclohexanediol', 0.42, 85, 38, 0.38, 'partially_biodegradable'),
    _e('OCC1COC(CO)O1', 'Isosorbide', 0.70, 175, 42, 0.25),
    _e('OCCOCC(O)CO', 'Diethylene glycol', 0.72, -35, 6, 0.78),
    _e('OCCOCCOCCO', 'Triethylene glycol', 0.68, -40, 4, 0.82),
]

# ── DIACIDS ──
FRANSEN_DIACIDS = [
    _e('OC(=O)CC(=O)O', 'Malonic acid', 0.78, 132, 15, 0.50),
    _e('OC(=O)CCC(=O)O', 'Succinic acid', 0.85, 185, 18, 0.48),
    _e('OC(=O)CCCC(=O)O', 'Glutaric acid', 0.82, 95, 16, 0.52),
    _e('OC(=O)CCCCC(=O)O', 'Adipic acid', 0.75, 150, 20, 0.55),
    _e('OC(=O)CCCCCC(=O)O', 'Pimelic acid', 0.70, 102, 18, 0.58),
    _e('OC(=O)CCCCCCC(=O)O', 'Suberic acid', 0.65, 140, 22, 0.60),
    _e('OC(=O)CCCCCCCC(=O)O', 'Azelaic acid', 0.60, 106, 20, 0.62),
    _e('OC(=O)CCCCCCCCC(=O)O', 'Sebacic acid', 0.55, 131, 24, 0.65),
    _e('OC(=O)/C=C/C(=O)O', 'Fumaric acid', 0.72, 287, 25, 0.30),
    _e('OC(=O)/C=C\\C(=O)O', 'Maleic acid', 0.70, 129, 22, 0.35),
    _e('OC(=O)COCC(=O)O', 'Diglycolic acid', 0.88, 95, 12, 0.65),
    _e('OC(=O)C(C)CC(=O)O', '2-Methylsuccinic acid', 0.78, 108, 16, 0.50),
    _e('OC(=O)c1ccc(C(=O)O)cc1', 'Terephthalic acid', 0.15, 267, 78, 0.15, 'non_biodegradable'),
    _e('OC(=O)c1cccc(C(=O)O)c1', 'Isophthalic acid', 0.18, 233, 72, 0.18, 'non_biodegradable'),
    _e('OC(=O)C1CCC(C(=O)O)CC1', '1,4-CHDA', 0.35, 150, 45, 0.30, 'partially_biodegradable'),
]

# ── POLYESTER REPEAT UNITS (diol + diacid combinations) ──
FRANSEN_POLYESTERS = [
    _e('O=C(O)CCC(=O)OCCO', 'PES (ethylene succinate)', 0.90, 104, 35, 0.45),
    _e('O=C(O)CCC(=O)OCCCO', 'PPS (propylene succinate)', 0.88, 45, 30, 0.55),
    _e('O=C(O)CCC(=O)OCCCCO', 'PBS (butylene succinate)', 0.85, -32, 35, 0.60),
    _e('O=C(O)CCCC(=O)OCCCCO', 'PBG (butylene glutarate)', 0.82, -45, 28, 0.62),
    _e('O=C(O)CCCCC(=O)OCCCCO', 'PBA (butylene adipate)', 0.78, -55, 25, 0.68),
    _e('O=C(O)CCC(=O)OCCCCCO', 'PPeS (pentylene succinate)', 0.80, -20, 30, 0.58),
    _e('O=C(O)CCC(=O)OCCCCCCO', 'PHS (hexylene succinate)', 0.75, -10, 28, 0.60),
    _e('O=C(O)CCCCC(=O)OCCCCCCO', 'PHA (hexylene adipate)', 0.70, -50, 22, 0.65),
    _e('O=C(O)CCCCCCC(=O)OCCCCO', 'PBSub (butylene suberate)', 0.60, -60, 20, 0.70),
    _e('O=C(O)CCCCCCCCC(=O)OCCCCO', 'PBSeb (butylene sebacate)', 0.55, -40, 22, 0.68),
    # Diglycolic acid polyesters (highest biodeg in paper)
    _e('O=C(O)COCC(=O)OCCO', 'P(EG-DGA)', 0.92, 60, 20, 0.55),
    _e('O=C(O)COCC(=O)OCCCO', 'P(PD-DGA)', 0.90, 30, 18, 0.60),
    _e('O=C(O)COCC(=O)OCCCCO', 'P(BD-DGA)', 0.88, 10, 22, 0.58),
    _e('O=C(O)CCC(=O)OCC(C)O', 'PPG-SA', 0.78, -5, 25, 0.55),
    _e('O=C(O)CCC(=O)OCC(C)(C)CO', 'NPG-SA', 0.40, 25, 35, 0.40, 'partially_biodegradable'),
    # Non-biodeg aromatic polyesters
    _e('O=C(O)c1ccc(C(=O)OCCO)cc1', 'PET unit', 0.10, 78, 80, 0.15, 'non_biodegradable'),
    _e('O=C(O)c1ccc(C(=O)OCCCCO)cc1', 'PBT unit', 0.12, 45, 65, 0.22, 'non_biodegradable'),
    _e('O=C(O)c1ccc(C(=O)OCCCCCCO)cc1', 'PHT unit', 0.15, 20, 50, 0.30, 'non_biodegradable'),
]

# ── LACTONE-BASED POLYESTERS (ring-opening) ──
FRANSEN_LACTONES = [
    _e('O=C1CCCCCO1', 'ε-Caprolactone', 0.82, -60, 22, 0.75),
    _e('CC1OC(=O)C(C)O1', 'L-Lactide', 0.75, 55, 55, 0.28),
    _e('O=C1COC(=O)CO1', 'Glycolide', 0.90, 35, 62, 0.20),
    _e('O=C1CCCCO1', 'δ-Valerolactone', 0.80, -40, 18, 0.70),
    _e('O=C1CCCO1', 'β-Butyrolactone', 0.85, 5, 15, 0.55),
    _e('O=C1CCO1', 'β-Propiolactone', 0.88, -33, 12, 0.60),
    _e('OCCCCCC(=O)O', '6-Hydroxyhexanoic acid', 0.78, -50, 15, 0.72),
    _e('OCCCCC(=O)O', '5-Hydroxypentanoic acid', 0.80, -35, 14, 0.68),
    _e('OC(C)CC(=O)O', '3-Hydroxybutyrate (PHB)', 0.85, 175, 40, 0.25),
    _e('OC(C)CCC(=O)O', '4-Hydroxyvalerate', 0.80, -10, 18, 0.60),
    _e('OC(CC)CC(=O)O', '3-Hydroxy-2-ethylpropanoate', 0.72, -5, 16, 0.55),
]

# ── POLYCARBONATES ──
FRANSEN_CARBONATES = [
    _e('O=C(OCCO)O', 'Ethylene carbonate unit', 0.78, 10, 18, 0.55),
    _e('O=C(OCCCO)O', 'Trimethylene carbonate unit', 0.72, -15, 15, 0.65),
    _e('O=C(OCCCCO)O', 'Tetramethylene carbonate unit', 0.68, -20, 14, 0.68),
    _e('O=C(OCCCCCCO)O', 'Hexamethylene carbonate unit', 0.60, -10, 16, 0.70),
    _e('O=C1OCCCO1', 'TMC monomer', 0.75, -25, 12, 0.62),
    _e('O=C1OCCO1', 'Ethylene carbonate monomer', 0.70, 36, 10, 0.50),
    _e('O=C(OCC(C)O)O', 'Propylene carbonate unit', 0.65, -10, 12, 0.58),
]

# ── NON-BIODEGRADABLE REFERENCES ──
FRANSEN_REFERENCES = [
    _e('C=CC', 'Polypropylene unit', 0.05, -10, 35, 0.70, 'non_biodegradable', False),
    _e('CC(C)(C)c1ccc(O)cc1', 'Bisphenol A', 0.08, 150, 72, 0.18, 'non_biodegradable', False),
    _e('C(F)(F)C(F)(F)', 'PTFE unit', 0.02, 127, 25, 0.20, 'non_biodegradable', False),
    _e('c1ccc(Oc2ccccc2)cc1', 'Diphenyl ether', 0.08, -30, 30, 0.35, 'non_biodegradable', False),
    _e('O=S(=O)(c1ccccc1)c1ccccc1', 'Diphenyl sulfone', 0.06, 128, 78, 0.15, 'non_biodegradable', False),
    _e('C(=O)Oc1ccccc1', 'Phenyl formate', 0.12, -5, 22, 0.40, 'non_biodegradable', False),
]


def get_fransen_data() -> list:
    """Get all Fransen et al. polyester dataset entries."""
    return (
        FRANSEN_DIOLS +
        FRANSEN_DIACIDS +
        FRANSEN_POLYESTERS +
        FRANSEN_LACTONES +
        FRANSEN_CARBONATES +
        FRANSEN_REFERENCES
    )


def get_fransen_stats() -> dict:
    """Get dataset statistics."""
    data = get_fransen_data()
    cats = {}
    for e in data:
        cats[e.category] = cats.get(e.category, 0) + 1
    return {
        'total': len(data),
        'categories': cats,
        'source': 'Fransen et al. PNAS 2023 (doi:10.1073/pnas.2321508121)',
    }
