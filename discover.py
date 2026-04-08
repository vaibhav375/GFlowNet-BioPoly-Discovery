#!/usr/bin/env python3
"""
🧪 GFlowNet Polymer Discovery — Quick Start
=============================================
Discover sustainable biodegradable alternatives to any plastic.

Usage:
    python discover.py polyethylene
    python discover.py polypropylene --top-k 30
    python discover.py PET --num-candidates 1000
    python discover.py "C=C" --output-dir ./my_results
    
Known polymers:
    polyethylene (PE), polypropylene (PP), polystyrene (PS),
    pvc, pet, nylon, abs, pla, pcl, pga, pbs, pha

You can also input any valid SMILES string.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from discovery.polymer_discovery import main

if __name__ == "__main__":
    main()
