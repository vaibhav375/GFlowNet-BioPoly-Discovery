# 🌱 GFlowNet-BioPoly-Discovery

**Multi-Objective Generative Flow Networks for Stochastic Discovery of Biodegradable Polymer Alternatives**

A Green AI system that discovers sustainable, biodegradable alternatives to conventional plastics by simultaneously optimizing for biodegradability, mechanical performance, and synthesizability — with minimal computational footprint.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Installation and Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Pipeline Phases](#-pipeline-phases)
- [Interactive UI](#-interactive-ui)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🔬 Project Overview

This project uses **Generative Flow Networks (GFlowNets)** to discover biodegradable polymer alternatives to conventional plastics (PET, PE, PP, PS, etc.). The system:

1. Trains surrogate GNN models to predict biodegradability and mechanical properties
2. Uses GFlowNet with trajectory balance loss to explore the polymer chemical space
3. Applies active learning to iteratively improve discovery quality
4. Outputs ranked biodegradable alternatives with 2D molecular structures

**Key achievements:**
- 918x faster degradation than PET (top candidate BP-001: 4.9 months vs 450+ years)
- 100% chemical validity and 99.96% novelty in generated molecules
- Only 0.643 kg CO2 for entire pipeline (CPU-only, no GPU required)
- Grade A validation: 13/13 tests passed

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset Size | 8,922 unique polymer structures |
| Generated Candidates | 4,624 (100% valid, 99.96% novel) |
| Diversity (Tanimoto) | 0.805 |
| Top Candidate | BP-001: 918x faster degradation, 30.4 MPa tensile |
| Active Learning | 12 rounds, 84.2% cumulative improvement |
| Carbon Footprint | 0.643 kg CO2 (US grid) |
| Model Size | 1.6M parameters (6.48 MB) |
| Runtime | ~21.4 hours on Apple M-series CPU |

---

## 📁 Repository Structure

```
GFlowNet-BioPoly-Discovery/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # ⭐ Main entry point — full 7-phase pipeline
├── discover.py                  # Standalone discovery script
├── run_demo.py                  # Quick demo with pretrained models
├── run_ablation.py              # Ablation study runner
├── validate_pipeline.py         # Validation suite (13/13 tests)
├── generate_presentation.py     # Generate presentation slides
├── generate_presentation_pdf.py # Generate PDF presentation
│
├── configs/                     # Configuration files
│   └── pipeline_config.py       # All hyperparameters and settings
│
├── models/                      # Neural network models
│   ├── gflownet.py              # GFlowNet core (1.6M params)
│   ├── policy_network.py        # 5-layer GINE policy network
│   ├── surrogate_bio.py         # Biodegradability MPNN (773K params)
│   ├── surrogate_mech.py        # Mechanical properties MPNN (841K params)
│   ├── surrogate_syn.py         # Synthesizability scorer
│   ├── mogfn.py                 # Multi-objective GFlowNet
│   ├── advanced_training.py     # LS-GFN, Thompson sampling, RLOO
│   └── genetic_refinement.py    # Genetic algorithm post-refinement
│
├── data/                        # Dataset and preprocessing
│   ├── preprocessing.py         # SMILES to molecular graph conversion
│   ├── polymer_smiles_db.py     # 126 curated polymer SMILES
│   ├── real_polymer_data.py     # 752 experimentally validated entries
│   ├── fransen_polyester_data.py# 73 biodegradable polyester structures
│   └── new_real_data/           # Raw CSV/TSV data files
│
├── training/                    # Training scripts
│   ├── train_surrogates.py      # Surrogate model training loop
│   └── active_learning.py       # 12-round active learning loop
│
├── evaluation/                  # Evaluation and metrics
│   ├── metrics.py               # Diversity, validity, novelty metrics
│   ├── green_ai_metrics.py      # Carbon footprint tracking
│   └── visualization.py         # Result visualization
│
├── discovery/                   # Polymer discovery engine
│   └── polymer_discovery.py     # Alternative discovery and ranking
│
├── simulation/                  # Molecular dynamics
│   └── md_simulation.py         # UFF MD validation simulator
│
├── checkpoints/                 # Trained model weights (auto-generated)
│   ├── s_bio_best.pt            # Best biodegradability surrogate
│   ├── s_mech_best.pt           # Best mechanical surrogate
│   ├── gflownet_best.pt         # Best GFlowNet policy
│   └── pipeline_config.json     # Saved hyperparameters
│
├── results/                     # Outputs (auto-generated)
│   ├── paper_results.json       # Full results JSON
│   ├── green_ai_report.json     # Sustainability metrics
│   └── discovery/               # Discovery output figures
│
├── ui/                          # Interactive interfaces
│   ├── index.html               # Static web UI (no install needed)
│   ├── app.py                   # Gradio Python interface
│   ├── app.js                   # JavaScript logic
│   ├── style.css                # Styling
│   └── mol_images/              # SVG molecular structure images
│
├── paper/                       # Research paper assets
│   └── main_restructured.tex    # LaTeX source
│
└── reproducibility/             # Reproducibility documentation
    ├── hyperparameters.txt      # Complete hyperparameter specs
    ├── random_seeds.txt         # All random seeds used
    ├── hardware_specs.txt       # Hardware requirements
    └── evaluation_guide.txt     # Step-by-step evaluation guide
```

---

## ✅ Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4-core, 2.5 GHz | 8-core (Apple M-series, Intel i7, AMD Ryzen 7) |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB SSD |
| GPU | Not required | Not required (CPU-only) |

### Software Requirements

- **Python 3.10+**
- **pip** or **conda** package manager
- **Git**

> No GPU required. The project runs entirely on CPU.

---

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/vaibhav375/GFlowNet-BioPoly-Discovery.git
cd GFlowNet-BioPoly-Discovery
```

### Step 2: Create a Virtual Environment

Using venv:
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate          # Windows
```

Using conda:
```bash
conda create -n gflownet-biopoly python=3.10
conda activate gflownet-biopoly
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with torch-scatter or torch-sparse:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

If rdkit is missing:
```bash
pip install rdkit
# Or: conda install -c conda-forge rdkit
```

### Step 4: Verify Installation

```bash
python validate_pipeline.py
```

Expected: **Grade A: 13/13 tests passed**

---

## ▶️ Running the Project

### Quick Demo (Recommended for First Run, ~2-3 hours)

```bash
python run_pipeline.py PET --quick
```

### Full Pipeline (~21 hours on Apple M-series)

```bash
python run_pipeline.py PET
```

### Discover Alternatives for Other Plastics

```bash
python run_pipeline.py PE       # Polyethylene
python run_pipeline.py PP       # Polypropylene
python run_pipeline.py PS       # Polystyrene
python run_pipeline.py nylon    # Nylon
python run_pipeline.py PVC      # Polyvinyl Chloride
```

### Resume from a Specific Phase (if interrupted)

```bash
python run_pipeline.py PET --resume-from 4   # Skip phases 1-3, use checkpoints
python run_pipeline.py PET --resume-from 6   # Skip to discovery phase
```

### Custom Parameters

```bash
python run_pipeline.py PET \
  --steps 2000 \           # GFlowNet training steps
  --top-k 15 \             # Number of top alternatives to report
  --candidates 1000 \      # Number of molecules to generate
  --dataset-size 5000 \    # Training dataset size
  --surrogate-epochs 100    # Surrogate model training epochs
```

### Run with a Custom Config File

```bash
python run_pipeline.py PET --config configs/my_config.json
```

---

## 🔄 Pipeline Phases

| Phase | Name | Description | Approx. Time |
|-------|------|-------------|-------------|
| 1 | Data Preparation | Generate 8,922 polymer SMILES, train/val/test split | ~1 min |
| 2 | Surrogate Training | Train S_bio and S_mech GNN models | ~2.5 hrs |
| 3 | GFlowNet Training | Train policy network with trajectory balance | ~15 hrs |
| 4 | Molecule Generation | Generate and evaluate 4,624 candidates | ~15 min |
| 5 | Active Learning | 12 rounds of generate, simulate, retrain | ~3.5 hrs |
| 6 | Discovery | Find and rank alternatives for target plastic | ~2 min |
| 7 | Report Generation | Export JSON results and Green AI report | ~1 min |

---

## 🖥️ Interactive UI

### Static HTML Interface (No Installation Needed)

```bash
open ui/index.html        # macOS
xdg-open ui/index.html   # Linux
start ui/index.html       # Windows
```

### Gradio Python Interface

```bash
pip install gradio
python ui/app.py
# Open: http://localhost:7860
```

---

## ⚙️ Configuration

Key settings in `configs/pipeline_config.py`:

```python
# Dataset
data.dataset_size = 8000         # Number of training molecules

# Surrogate Models
surrogate.epochs = 150           # Max training epochs
surrogate.hidden_dim = 256       # GNN hidden dimension

# GFlowNet
gflownet.training_steps = 2000   # Training iterations
gflownet.alpha_bio = 0.50        # Biodegradability reward weight
gflownet.alpha_mech = 0.35       # Mechanical reward weight
gflownet.alpha_syn = 0.15        # Synthesizability reward weight

# Discovery
discovery.top_k = 20             # Number of top alternatives
```

---

## 📤 Output Files

After running, results are saved in:

```
results/
├── paper_results.json       # Complete results (metrics, top candidates)
├── pipeline_config.json     # Hyperparameters used
├── green_ai_report.json     # Carbon footprint and energy metrics
├── pipeline_log.txt         # Full run log
└── discovery/               # Top candidate visualizations

checkpoints/
├── s_bio_best.pt            # Best biodegradability model weights
├── s_mech_best.pt           # Best mechanical model weights
└── gflownet_best.pt         # Best GFlowNet weights
```

---

## 🛠️ Troubleshooting

**torch_geometric not found:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

**rdkit not found:**
```bash
pip install rdkit
```

**Out of memory:**
```bash
python run_pipeline.py PET --quick --dataset-size 2000
```

**Pipeline interrupted:**
```bash
python run_pipeline.py PET --resume-from 4
```

---

## 🌍 Green AI Metrics

| Phase | Time (h) | Energy (kWh) | CO2 US (kg) |
|-------|----------|--------------|-------------|
| Data prep | 0.01 | 0.001 | 0.0004 |
| Surrogate training | 2.4 | 0.172 | 0.072 |
| GFlowNet training | 15.3 | 1.095 | 0.460 |
| Active learning | 3.4 | 0.243 | 0.102 |
| **Total** | **21.4** | **1.527** | **0.643** |

---

## 🎓 Citation

```bibtex
@article{handoo2024gflownet,
  title={Multi-Objective Generative Flow Networks for Stochastic Discovery 
         of Biodegradable Polymer Alternatives: A Green AI Approach},
  author={Handoo, Vaibhav},
  year={2026},
  note={Code: https://github.com/vaibhav375/GFlowNet-BioPoly-Discovery}
}
```

---

## 📞 Contact

**Vaibhav Handoo**
- Email: handoovaibhav123@gmail.com
- Institution: Department of Computer Science, PES University

---

## 📄 License

- Code: MIT License
- Data: CC-BY-4.0
- Paper: All rights reserved

---

*Made with love for a sustainable future*
