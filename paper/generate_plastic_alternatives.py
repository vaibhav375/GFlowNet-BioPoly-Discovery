#!/usr/bin/env python3
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

FIGDIR = 'paper/figures'
os.makedirs(FIGDIR, exist_ok=True)

with open('results/discovery/discovery_results.json') as f:
    discovery = json.load(f)

cands = discovery['candidates'][:30]

# Categorize candidates based on predicted properties
pe_alts = []   # PE: High flex
pet_alts = []  # PET: High tensile, high Tg
ps_alts = []   # PS: Low flex, high Tg
pla_alts = []  # PLA: Ultra-fast biodeg

for c in cands:
    t = c.get('predicted_tensile', 0)
    f = c.get('predicted_flexibility', 0)
    tg = c.get('predicted_tg', 0)
    b = c.get('predicted_biodeg_months', 0)
    
    if len(pe_alts) < 3 and f >= 0.8 and tg < 120:
        pe_alts.append(c)
    elif len(pet_alts) < 3 and t >= 30 and tg >= 140:
        pet_alts.append(c)
    elif len(ps_alts) < 3 and f <= 0.6 and tg >= 130:
        ps_alts.append(c)
    elif len(pla_alts) < 3 and b <= 5.2 and c not in pe_alts and c not in pet_alts and c not in ps_alts:
        pla_alts.append(c)

# Fallbacks if strict criteria not met
for c in cands:
    if len(pe_alts) < 3 and c not in pe_alts+pet_alts+ps_alts+pla_alts: pe_alts.append(c)
    if len(pet_alts) < 3 and c not in pe_alts+pet_alts+ps_alts+pla_alts: pet_alts.append(c)
    if len(ps_alts) < 3 and c not in pe_alts+pet_alts+ps_alts+pla_alts: ps_alts.append(c)
    if len(pla_alts) < 3 and c not in pe_alts+pet_alts+ps_alts+pla_alts: pla_alts.append(c)

def create_grid(group, title_prefix):
    mols = []
    labels = []
    for c in group:
        m = Chem.MolFromSmiles(c['smiles'])
        mols.append(m)
        labels.append(f"{c['name']}\nBiodeg: {c['predicted_biodeg_months']:.1f}mo")
    return Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 250), legends=labels, returnPNG=False)

img_pe = create_grid(pe_alts, "PE")
img_pet = create_grid(pet_alts, "PET")
img_ps = create_grid(ps_alts, "PS")
img_pla = create_grid(pla_alts, "PLA")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].imshow(img_pe)
axes[0,0].set_title("(a) Polyethylene (PE) Alternatives\nHigh flexibility for films/bags", fontsize=14, pad=10)
axes[0,0].axis('off')

axes[0,1].imshow(img_pet)
axes[0,1].set_title("(b) Polyethylene Terephthalate (PET) Alternatives\nHigh tensile strength for bottles/packaging", fontsize=14, pad=10)
axes[0,1].axis('off')

axes[1,0].imshow(img_ps)
axes[1,0].set_title("(c) Polystyrene (PS) Alternatives\nHigh rigidity for cutlery/cases", fontsize=14, pad=10)
axes[1,0].axis('off')

axes[1,1].imshow(img_pla)
axes[1,1].set_title("(d) Polylactic Acid (PLA) Alternatives\nUltra-fast biodegradation for compostables", fontsize=14, pad=10)
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig4_polymer_alternatives.png', dpi=300, bbox_inches='tight')
print("✅ Fig 4: Polymer alternatives saved")
