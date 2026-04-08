# 🌱 GFlowNet for Biodegradable Polymer Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Green AI](https://img.shields.io/badge/Green%20AI-0.643%20kg%20CO₂-green.svg)](https://github.com/yourusername/GFlowNet-BioPoly-Discovery)

> **Multi-Objective Generative Flow Networks for Stochastic Discovery of Biodegradable Polymer Alternatives: A Green AI Approach**

Discover sustainable, biodegradable alternatives to conventional plastics using state-of-the-art Generative Flow Networks (GFlowNets). This system simultaneously optimizes for biodegradability, mechanical performance, and synthesizability while maintaining minimal computational footprint.

---

## 🎯 Key Highlights

- **918× Faster Degradation:** Top candidate (BP-001) biodegrades in 4.9 months vs. PET's 450+ years
- **100% Validity:** All 4,624 generated candidates are chemically valid
- **99.96% Novelty:** Nearly all candidates are novel structures
- **Green AI:** Only 0.643 kg CO₂ (US grid) or 1.252 kg CO₂ (India grid) for entire pipeline
- **Real Data:** Trained on 752 experimentally validated polymers from PolyInfo and PN2S databases
- **CPU-Only:** Runs on consumer hardware without GPU (21.4 hours on Apple M-series)

---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| **Dataset Size** | 8,922 unique polymer structures |
| **Real Data Points** | 752 experimentally validated entries |
| **Generated Candidates** | 4,624 (100% valid, 99.96% novel) |
| **Diversity** | 0.805 (Tanimoto) |
| **Top Candidate** | BP-001: 918× faster degradation, 30.4 MPa tensile |
| **Active Learning** | 12 rounds, 84.2% cumulative improvement |
| **Carbon Footprint** | 0.643 kg CO₂ (US) / 1.252 kg CO₂ (India) |
| **Energy** | 1.527 kWh (21.4 hours runtime) |
| **Model Size** | 1.6M parameters (6.48 MB) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GFlowNet-BioPoly-Discovery.git
cd GFlowNet-BioPoly-Discovery

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Quick demo (uses pretrained models)
python run_pipeline.py PET --quick

# Full pipeline (trains from scratch)
python run_pipeline.py PET

# Discover alternatives for other plastics
python run_pipeline.py PE
python run_pipeline.py PP
python run_pipeline.py PS
```

### Launch Interactive UI

```bash
# Static HTML interface (no installation needed)
open ui/index.html

# Or use Python Gradio interface
python ui/app.py
# Navigate to http://localhost:7860
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    7-Phase Discovery Pipeline                    │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Data Preparation
  ├─ 8,922 polymer SMILES → molecular graphs
  ├─ 752 real data (2× weighted) + 126 curated + augmentation
  └─ 38-dim atom features, 7-dim bond features

Phase 2: GNN Surrogate Training
  ├─ S_bio: Biodegradability predictor (773K params, epoch 69)
  ├─ S_mech: Mechanical properties (841K params, epoch 73)
  └─ S_syn: Synthesizability scorer (rule-based SA Score)

Phase 3: GFlowNet Training
  ├─ 5-layer GINE policy network (1.6M params)
  ├─ Trajectory Balance + Sub-TB + RLOO losses
  ├─ 8,000 training steps with curriculum learning
  └─ Multi-objective reward: R = S_bio^0.50 × S_mech^0.35 × S_syn^0.15

Phase 4: Molecule Generation
  ├─ Generate 4,624 candidates
  ├─ 100% validity, 99.96% novelty, 0.805 diversity
  └─ Chemistry-aware reward shaping (28 bonuses + 13 penalties)

Phase 5: Active Learning
  ├─ 12 rounds of generate → simulate → retrain
  ├─ UCB acquisition (β=1.35 → 0.3)
  ├─ UFF molecular dynamics validation (960 molecules)
  └─ 84.2% cumulative improvement

Phase 6: Discovery & Ranking
  ├─ Extract Pareto front
  ├─ Rank by multi-objective reward
  └─ Filter by polymerizability and stability

Phase 7: Green AI Report
  ├─ Track CO₂, energy, water footprint
  ├─ 0.643 kg CO₂ (US) / 1.252 kg CO₂ (India)
  └─ Sustainability score: 88.4/100
```

---

## 📁 Project Structure

```
GFlowNet-BioPoly-Discovery/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 run_pipeline.py              # Main pipeline orchestrator
├── 📄 discover.py                  # Discovery script
├── 📄 validate_pipeline.py         # Validation (Grade A: 13/13 tests)
│
├── 📂 models/                      # Neural network models
│   ├── gflownet.py                 # GFlowNet core (1.6M params)
│   ├── policy_network.py           # 5-layer GINE policy
│   ├── surrogate_bio.py            # Biodegradability MPNN (773K)
│   ├── surrogate_mech.py           # Mechanical properties MPNN (841K)
│   ├── surrogate_syn.py            # Synthesizability scorer
│   ├── mogfn.py                    # Multi-objective GFlowNet
│   ├── advanced_training.py        # LS-GFN, Thompson sampling
│   └── genetic_refinement.py       # Genetic algorithm refinement
│
├── 📂 data/                        # Dataset and preprocessing
│   ├── preprocessing.py            # SMILES → graph conversion
│   ├── polymer_smiles_db.py        # 126 curated polymers
│   ├── real_polymer_data.py        # 752 validated entries
│   ├── fransen_polyester_data.py   # 73 polyester structures
│   └── new_real_data/              # Raw data files (CSV, TSV)
│
├── 📂 training/                    # Training scripts
│   ├── train_surrogates.py         # Surrogate model training
│   └── active_learning.py          # 12-round active learning
│
├── 📂 evaluation/                  # Metrics and visualization
│   ├── metrics.py                  # Diversity, validity, novelty
│   ├── green_ai_metrics.py         # Carbon footprint tracking
│   └── visualization.py            # Result visualization
│
├── 📂 discovery/                   # Polymer discovery engine
│   └── polymer_discovery.py        # Alternative discovery
│
├── 📂 checkpoints/                 # Trained model weights
│   ├── s_bio_best.pt               # Best biodeg surrogate
│   ├── s_mech_best.pt              # Best mechanical surrogate
│   ├── gflownet_best.pt            # Best GFlowNet policy
│   └── pipeline_config.json        # Hyperparameters
│
├── 📂 results/                     # Experimental results
│   ├── paper_results.json          # Main results
│   ├── discovery/                  # Discovery outputs
│   ├── active_learning/            # AL history
│   └── green_ai_report.json        # Sustainability metrics
│
├── 📂 ui/                          # Interactive interfaces
│   ├── index.html                  # Static web interface
│   ├── app.py                      # Gradio interface
│   ├── app.js                      # JavaScript logic
│   ├── style.css                   # Styling
│   ├── mol_images/                 # SVG structures (21 files)
│   └── README.md                   # UI documentation
│
├── 📂 paper/                       # Research paper
│   ├── main_restructured.tex       # LaTeX source
│   ├── figures/                    # Generated figures (11 files)
│
└── 📂 reproducibility/             # Reproducibility docs
    ├── README.md                   # Overview
    ├── hyperparameters.txt         # Complete specs
    ├── random_seeds.txt            # All seeds
    ├── hardware_specs.txt          # Hardware info
    └── evaluation_guide.txt        # Step-by-step guide
```

---

## 🧪 Key Features

### Multi-Objective Optimization
- **Biodegradability:** Ester/amide bonds, lactone rings, O:C ratio
- **Mechanical Properties:** Tensile strength, glass transition, flexibility
- **Synthesizability:** SA Score-based feasibility

### Advanced Training Techniques
- **Trajectory Balance (TB)** + **Sub-Trajectory Balance (SubTB)**
- **REINFORCE Leave-One-Out (RLOO)** variance reduction
- **Entropy regularization** for exploration
- **EMA weight tracking** for stable inference
- **Progressive curriculum** learning (8 → 15 → 25 atoms)
- **Reward-prioritized replay buffer** (2000 capacity)

### Chemistry-Aware Reward Shaping
- **28 Positive Bonuses:** Ester (+0.15), lactone (+0.35), amide (+0.10), etc.
- **13 Negative Penalties:** Peroxide (-0.80), exotic atoms (-0.50), etc.
- **Adaptive exponents:** S_bio increases from 0.50 → 0.55 during training
- **Hard thresholds:** S_bio ≥ 0.6, S_mech ≥ 0.5, S_syn ≥ 0.5

### Active Learning
- **12 rounds** of iterative improvement
- **UCB acquisition** (β: 1.5 → 0.3)
- **MC Dropout** uncertainty (K=10 passes)
- **UFF molecular dynamics** validation
- **84.2% cumulative improvement**

---

## 📈 Performance

### Generation Quality
| Metric | Value |
|--------|-------|
| Validity | 100% |
| Uniqueness | 100% |
| Novelty | 99.96% |
| Diversity (Tanimoto) | 0.805 |
| Mean Reward | 0.569 ± 0.274 |
| Best Reward | 1.380 |
| Top-10 Avg Reward | 1.356 |

### Surrogate Models
| Model | Parameters | Best Epoch | Val Loss |
|-------|-----------|------------|----------|
| S_bio | 773,761 | 69/300 | 0.00414 |
| S_mech | 841,363 | 73/300 | 0.00931 |
| S_syn | Rule-based | - | - |

**Compute Savings:** 75% (early stopping at epochs 69/73 vs. 300 max)

### Top Candidates

| Rank | Name | Biodeg (months) | Improvement | Tensile (MPa) | Reward |
|------|------|-----------------|-------------|---------------|--------|
| 1 | BP-001 | 4.9 | 918× | 30.4 | 1.336 |
| 2 | BP-007 | 5.3 | 849× | 31.0 | 1.301 |
| 3 | BP-002 | 5.3 | 849× | 29.0 | 1.289 |
| 4 | BP-004 | 6.1 | 738× | 27.7 | 1.274 |
| 5 | BP-005 | 6.3 | 714× | 28.2 | 1.273 |

---

## 🌍 Green AI Metrics

### Carbon Footprint

| Phase | Time (h) | Energy (kWh) | CO₂ US (kg) | CO₂ India (kg) |
|-------|----------|--------------|-------------|----------------|
| Data prep | 0.01 | 0.001 | 0.0004 | 0.0008 |
| Surrogate training | 2.4 | 0.172 | 0.072 | 0.141 |
| GFlowNet training | 15.3 | 1.095 | 0.460 | 0.898 |
| Active learning | 3.4 | 0.243 | 0.102 | 0.199 |
| Evaluation | 0.2 | 0.014 | 0.006 | 0.011 |
| Discovery | 0.03 | 0.002 | 0.001 | 0.002 |
| **Total** | **21.4** | **1.527** | **0.643** | **1.252** |

**Grid Carbon Intensity:**
- US: 0.42 kg CO₂/kWh
- India: 0.82 kg CO₂/kWh

### Efficiency Metrics
- **Molecules per kWh:** 3,022
- **Carbon efficiency:** 7,195 molecules/kg CO₂ (US grid)
- **FLOPs per molecule:** 269 million
- **Parameters per valid molecule:** 350.6
- **Sustainability score:** 88.4/100

### Comparison with Baselines
| Method | CO₂ (kg) | Reduction |
|--------|----------|-----------|
| Junction Tree VAE | 2.5 | 74% less |
| Genetic Algorithm | 1.8 | 64% less |
| PPO (RL) | 1.5 | 57% less |
| **Our GFlowNet** | **0.643** | **baseline** |

---

## 🎨 Interactive UI

### Static HTML Interface
Beautiful, fast, no-installation-required interface.

**Features:**
- Real-time polymer alternative discovery
- Interactive 2D molecular structures
- Multi-objective score radar charts
- Biodegradation time comparison
- Environmental impact visualization
- 7-phase pipeline with tooltips
- Green AI metrics dashboard

**Usage:**
```bash
open ui/index.html
```

### Gradio Python Interface
Dynamic interface with live model integration.

**Features:**
- Custom target polymer input
- Adjustable generation parameters
- Molecule explorer (analyze any SMILES)
- Training dashboard
- Green AI comparison charts
- Export functionality

**Usage:**
```bash
python ui/app.py
# Navigate to http://localhost:7860
```

---

## 🔬 Validation

### Grade A Validation (13/13 Tests Passed)

**Biodegradable Monomers (should score HIGH):**
- ✅ L-Lactic acid (PLA): S_bio=0.904, Reward=1.121
- ✅ Lactide (PLA dimer): S_bio=0.793, Reward=1.188
- ✅ ε-Caprolactone (PCL): S_bio=0.747, Reward=0.841
- ✅ Succinic acid (PBS): S_bio=0.840, Reward=1.173
- ✅ 3-HB acid (PHB): S_bio=0.889, Reward=1.120
- ✅ Glycolic acid (PGA): S_bio=0.966, Reward=1.104
- ✅ PBS dimer: S_bio=0.746, Reward=0.724
- ✅ Glucose (cellulose): S_bio=0.904, Reward=0.765

**Non-Biodegradable Plastics (should score LOW):**
- ✅ Decane (PE model): S_bio=0.110, Reward=0.015
- ✅ PS trimer: S_bio=0.090, Reward=0.040
- ✅ PET repeat unit: S_bio=0.089, Reward=0.073
- ✅ PVC oligomer: S_bio=0.094, Reward=0.001
- ✅ PP oligomer: S_bio=0.132, Reward=0.023

**Separation Ratio:** 33.19× (biodegradable avg / non-biodegradable avg)

---

## 📚 Documentation

### For Users
- **README.md** (this file) - Project overview
- **ui/README.md** - Interactive interface guide
- **reproducibility/README.md** - Reproducibility overview

### For Developers
- **reproducibility/hyperparameters.txt** - Complete specifications
- **reproducibility/random_seeds.txt** - All random seeds
- **reproducibility/hardware_specs.txt** - Hardware requirements
- **reproducibility/evaluation_guide.txt** - Step-by-step evaluation

### For Researchers
- **paper/main_restructured.tex** - Full research paper
- **paper/research_paper_final.pdf** - Compiled PDF
- **results/paper_results.json** - Complete results

---

## 🔧 Requirements

### Minimum
- **CPU:** 4-core (≥2.5 GHz)
- **RAM:** 8 GB
- **Storage:** 10 GB
- **GPU:** Not required (CPU-only)

### Recommended
- **CPU:** 8-core (Apple M-series, Intel i7, AMD Ryzen 7)
- **RAM:** 16 GB
- **Storage:** 20 GB SSD
- **GPU:** Not required

### Software
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- RDKit 2023.03+
- NumPy, Matplotlib, Gradio

See `requirements.txt` for complete list.

---

## 🚦 Usage Examples

### Discover Alternatives for PET
```python
from discovery.polymer_discovery import PolymerDiscoveryEngine

# Initialize engine
engine = PolymerDiscoveryEngine(checkpoint_dir='checkpoints/')
engine.load_models()

# Discover alternatives
result = engine.discover_alternatives(
    target_polymer='PET',
    num_candidates=1000,
    top_k=20
)

# Print results
for candidate in result.candidates:
    print(f"{candidate.name}: {candidate.biodeg_improvement_factor:.0f}× faster")
```

### Generate Custom Molecules
```python
from models.gflownet import GFlowNet

# Load trained model
gfn = GFlowNet.load_from_checkpoint('checkpoints/gflownet_best.pt')

# Generate molecules
molecules = gfn.generate(
    num_samples=100,
    temperature=1.0,
    max_atoms=20
)

# Evaluate
for mol in molecules:
    print(f"SMILES: {mol.smiles}, Reward: {mol.reward:.4f}")
```

### Evaluate Biodegradability
```python
from models.surrogate_bio import BiodegradabilitySurrogate

# Load model
s_bio = BiodegradabilitySurrogate.load('checkpoints/s_bio_best.pt')

# Predict
smiles = "CC(O)C(=O)OC(C)C(=O)O"  # Lactic acid dimer
score = s_bio.predict(smiles)
print(f"Biodegradability score: {score:.3f}")
```

---

## 🧬 Supported Plastics

The system can discover alternatives for:

| Plastic | Abbr | Common Uses | Degradation Time |
|---------|------|-------------|------------------|
| Polyethylene Terephthalate | PET | Water bottles, food containers | 450+ years |
| Polyethylene | PE | Plastic bags, bottles | 500+ years |
| Polypropylene | PP | Food packaging, bottle caps | 400+ years |
| Polystyrene | PS | Cups, takeaway containers | 500+ years |
| Polyvinyl Chloride | PVC | Pipes, window frames | 1000+ years |
| Nylon | Nylon | Textiles, fishing nets | 200+ years |
| Low-Density PE | LDPE | Cling wrap, garbage bags | 500+ years |
| High-Density PE | HDPE | Milk jugs, detergent bottles | 500+ years |

---

## 🎓 Citation

If you use this code or data, please cite:

```bibtex
@article{handoo2024gflownet,
  title={Multi-Objective Generative Flow Networks for Stochastic Discovery of Biodegradable Polymer Alternatives: A Green AI Approach},
  author={Handoo, Vaibhav},
  year={2026},
  note={Code available at: https://github.com/yourusername/GFlowNet-BioPoly-Discovery}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

- **Code:** MIT License
- **Data:** CC-BY-4.0
- **Paper:** All rights reserved

See `LICENSE` for details.

---

## 🌟 Acknowledgments

### Datasets
- **PolyInfo Database** (NIMS, Japan) - 752 experimentally validated polymers
- **PN2S Database** - 8,127 canonical SMILES with degradability rankings
- **Fransen et al. PNAS 2023** - 73 high-throughput biodegradable polyesters

### Frameworks
- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural networks
- **RDKit** - Cheminformatics toolkit
- **Gradio** - Interactive UI framework

### Inspiration
- **GFlowNet** (Bengio et al., 2021) - Flow-based generative models
- **MOGFN** (Jain et al., 2023) - Multi-objective GFlowNets
- **Sub-TB** (Madan et al., 2023) - Sub-trajectory balance

---

## 📞 Contact

**Vaibhav Handoo**
- Email: handoovaibhav123@gmail.com
- Institution: Department of Computer Science, PES University

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email the author
- Check the documentation in `reproducibility/`

---

## 🔗 Links

- **Paper:** `paper/research_paper_final.pdf`
- **Interactive Demo:** `ui/index.html`
- **Documentation:** `reproducibility/README.md`
- **Results:** `results/paper_results.json`

---

## ⚡ Quick Commands

```bash
# Install
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py PET

# Validate
python validate_pipeline.py

# Launch UI
python ui/app.py

# Generate figures
python paper/generate_figures.py

# Run ablation study
python run_ablation.py
```

---

## 🎯 Project Goals

1. **Environmental Impact:** Replace persistent plastics with biodegradable alternatives
2. **Scientific Rigor:** Experimentally grounded, reproducible research
3. **Accessibility:** CPU-only, low-carbon, open-source
4. **Practicality:** Real-world feasible candidates with polymerizability
5. **Transparency:** Complete documentation and reproducibility

---

## 🏆 Achievements

- ✅ **918× faster degradation** than PET (top candidate)
- ✅ **100% validity** in generated molecules
- ✅ **99.96% novelty** (nearly all new structures)
- ✅ **0.643 kg CO₂** total emissions (US grid)
- ✅ **Grade A validation** (13/13 tests passed)
- ✅ **752 real data points** (largest GFlowNet polymer dataset)
- ✅ **CPU-only execution** (no GPU required)
- ✅ **Complete reproducibility** (all seeds, hyperparameters documented)

---

<div align="center">

**Made with ♻️ for a sustainable future**

[⭐ Star this repo](https://github.com/vaibhav375/GFlowNet-BioPoly-Discovery) | [🐛 Report Bug](https://github.com/yourusername/GFlowNet-BioPoly-Discovery/issues) | [💡 Request Feature](https://github.com/yourusername/GFlowNet-BioPoly-Discovery/issues)

</div>
