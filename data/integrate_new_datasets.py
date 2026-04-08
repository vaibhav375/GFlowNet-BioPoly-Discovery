"""
Session 10 — Integrate new real-world polymer datasets.

Sources:
  1. PN2S (CMDM-Lab): 52K lines of polymer SMILES mapped from PolyInfo
  2. PolyInfo degradability (tsudalab): 4,577 polymer degradability rankings
  3. PN2S test set: 254 verified polymer SMILES with names

Output: data/new_real_data/merged_polymer_smiles.csv
"""
import csv
import re
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from collections import OrderedDict

NEW_DATA_DIR = os.path.join(os.path.dirname(__file__), 'new_real_data')


def parse_pn2s_block_file(filepath: str) -> dict:
    """Parse PN2S block-format files (ID/Name/Polymer Smiles/Monomer Smiles)."""
    polymers = {}  # pid -> {name, polymer_smiles, monomer_smiles}
    
    if not os.path.exists(filepath):
        print(f"  ⚠️  {filepath} not found, skipping")
        return polymers
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current = {}
    for line in lines:
        line = line.strip()
        if line.startswith('ID:'):
            if current.get('id') and current.get('polymer_smiles'):
                polymers[current['id']] = current
            current = {'id': line[3:]}
        elif line.startswith('Name:'):
            current['name'] = line[5:]
        elif line.startswith('Polymer Smiles:'):
            current['polymer_smiles'] = line[15:]
        elif line.startswith('Monomer Smiles 1:'):
            current['monomer_smiles'] = line[17:].strip()
        elif line.startswith('Monomer Smiles 2:'):
            current['monomer_smiles_2'] = line[17:].strip()
    
    # Don't forget the last entry
    if current.get('id') and current.get('polymer_smiles'):
        polymers[current['id']] = current
    
    return polymers


def parse_pn2s_test_tsv(filepath: str) -> list:
    """Parse PN2S test set TSV (Name\\tSMILES\\tURL)."""
    results = []
    if not os.path.exists(filepath):
        print(f"  ⚠️  {filepath} not found, skipping")
        return results
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                name = parts[0].strip()
                smiles = parts[1].strip()
                if smiles and name:
                    results.append({'name': name, 'smiles': smiles})
    return results


def parse_polyinfo_degradability(filepath: str) -> dict:
    """Parse PolyInfo degradability CSV (PID, Degradability score)."""
    degradability = {}
    if not os.path.exists(filepath):
        print(f"  ⚠️  {filepath} not found, skipping")
        return degradability
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get('PID', '').strip()
            deg = row.get('Degradability', '').strip()
            if pid and deg:
                try:
                    degradability[pid] = float(deg)
                except ValueError:
                    pass
    return degradability


def psmiles_to_smiles(psmiles: str) -> str:
    """Convert polymer SMILES (*-notation) to standard SMILES for RDKit."""
    if not psmiles:
        return ''
    
    # Replace * with H to make it parseable by RDKit
    # This gives us the monomer/repeat unit structure
    smi = psmiles.replace('*', '[H]')
    
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # Try removing the wildcards entirely
        smi = psmiles.replace('*', '')
        mol = Chem.MolFromSmiles(smi)
    
    if mol is None:
        return ''
    
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return ''


def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid via RDKit."""
    if not smiles or len(smiles) < 2:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        n_atoms = mol.GetNumHeavyAtoms()
        return 3 <= n_atoms <= 50  # Reasonable polymer monomer range
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  Session 10 — Integrating New Real-World Polymer Datasets")
    print("=" * 60)
    
    all_entries = OrderedDict()  # canonical_smiles -> {name, source, degradability, ...}
    
    # ── 1. Parse PN2S All Training (polymer SMILES + names + monomers) ──
    print("\n📦 Parsing PN2S All Training Set...")
    pn2s_file = os.path.join(NEW_DATA_DIR, 'pn2s_all_train.txt')
    pn2s_data = parse_pn2s_block_file(pn2s_file)
    print(f"  Found {len(pn2s_data)} polymer entries")
    
    pn2s_valid = 0
    pn2s_pid_to_smiles = {}
    for pid, info in pn2s_data.items():
        # Try the polymer SMILES first
        polymer_smi = psmiles_to_smiles(info.get('polymer_smiles', ''))
        # If that fails, try monomer SMILES
        if not polymer_smi and info.get('monomer_smiles'):
            monomer_smi = info['monomer_smiles'].strip()
            if validate_smiles(monomer_smi):
                polymer_smi = Chem.MolToSmiles(Chem.MolFromSmiles(monomer_smi))
        
        if polymer_smi and validate_smiles(polymer_smi):
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(polymer_smi))
            if canonical not in all_entries:
                all_entries[canonical] = {
                    'smiles': canonical,
                    'name': info.get('name', ''),
                    'source': 'PN2S',
                    'pid': pid,
                }
                pn2s_valid += 1
            pn2s_pid_to_smiles[pid] = canonical
    
    print(f"  ✅ {pn2s_valid} valid unique SMILES extracted from PN2S")
    
    # ── 2. Parse PN2S Test Set (verified polymer SMILES) ──
    print("\n📦 Parsing PN2S Test Set...")
    test_file = os.path.join(NEW_DATA_DIR, 'pn2s_test.tsv')
    test_data = parse_pn2s_test_tsv(test_file)
    print(f"  Found {len(test_data)} test entries")
    
    test_valid = 0
    for entry in test_data:
        smi = psmiles_to_smiles(entry['smiles'])
        if smi and validate_smiles(smi):
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            if canonical not in all_entries:
                all_entries[canonical] = {
                    'smiles': canonical,
                    'name': entry['name'],
                    'source': 'PN2S_test',
                }
                test_valid += 1
    
    print(f"  ✅ {test_valid} new unique SMILES from PN2S test set")
    
    # ── 3. Link PolyInfo degradability scores ──
    print("\n📦 Linking PolyInfo degradability rankings...")
    deg_file = os.path.join(NEW_DATA_DIR, 'polyinfo_degradability.csv')
    degradability = parse_polyinfo_degradability(deg_file)
    print(f"  Found {len(degradability)} degradability scores")
    
    linked = 0
    for pid, deg_score in degradability.items():
        if pid in pn2s_pid_to_smiles:
            smi = pn2s_pid_to_smiles[pid]
            if smi in all_entries:
                all_entries[smi]['degradability'] = deg_score
                linked += 1
    
    print(f"  ✅ {linked} polymers linked with degradability scores")
    
    # ── 4. Parse PN2S Condensation subset (biodeg-relevant) ──
    print("\n📦 Parsing PN2S Condensation polymers (biodeg-relevant)...")
    cond_file = os.path.join(NEW_DATA_DIR, 'pn2s_condensation1.txt')
    cond_data = parse_pn2s_block_file(cond_file)
    print(f"  Found {len(cond_data)} condensation polymer entries")
    
    cond_valid = 0
    for pid, info in cond_data.items():
        polymer_smi = psmiles_to_smiles(info.get('polymer_smiles', ''))
        if not polymer_smi and info.get('monomer_smiles'):
            monomer_smi = info['monomer_smiles'].strip()
            if validate_smiles(monomer_smi):
                polymer_smi = Chem.MolToSmiles(Chem.MolFromSmiles(monomer_smi))
        
        if polymer_smi and validate_smiles(polymer_smi):
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(polymer_smi))
            if canonical not in all_entries:
                all_entries[canonical] = {
                    'smiles': canonical,
                    'name': info.get('name', ''),
                    'source': 'PN2S_condensation',
                    'pid': pid,
                }
                cond_valid += 1
    
    print(f"  ✅ {cond_valid} new unique condensation polymer SMILES")
    
    # ── 5. Write merged output ──
    output_file = os.path.join(NEW_DATA_DIR, 'merged_polymer_smiles.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'smiles', 'name', 'source', 'pid', 'degradability'
        ])
        writer.writeheader()
        for entry in all_entries.values():
            writer.writerow({
                'smiles': entry.get('smiles', ''),
                'name': entry.get('name', ''),
                'source': entry.get('source', ''),
                'pid': entry.get('pid', ''),
                'degradability': entry.get('degradability', ''),
            })
    
    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {len(all_entries)} unique valid polymer SMILES")
    print(f"  Output: {output_file}")
    print(f"{'=' * 60}")
    
    # Summary by source
    sources = {}
    for e in all_entries.values():
        src = e.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    
    print("\n  By source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")
    
    has_deg = sum(1 for e in all_entries.values() if e.get('degradability'))
    print(f"\n  With degradability score: {has_deg}")
    
    return len(all_entries)


if __name__ == '__main__':
    main()
