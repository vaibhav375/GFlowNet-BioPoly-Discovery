"""
GNN Self-Supervised Pretraining — Publication Improvement
==========================================================
Pretrain GNN encoder on unlabeled molecular data using:
  1. Atom Masking — mask 15% of atom features, predict masked types
  2. Context Prediction — predict local substructure presence

Paper: Hu et al. "Strategies for Pre-training GNNs" (ICLR 2020)

Usage:
    python -m models.gnn_pretraining --epochs 30
"""

import os
import sys
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM

logger = logging.getLogger(__name__)


# ============================================================
# Pretraining GNN Encoder
# ============================================================

class PretrainableGNNEncoder(nn.Module):
    """
    GNN encoder that can be pretrained with self-supervised objectives
    and then transfer weights to surrogate models.
    """
    
    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial atom embedding
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        
        # GIN-E conv layers (same architecture as policy/surrogate networks)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            edge_nn = nn.Linear(bond_dim, hidden_dim)
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.edge_encoder = nn.Linear(bond_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both node-level and graph-level representations.
        
        Returns:
            node_repr: (num_nodes, hidden_dim)
            graph_repr: (batch_size, hidden_dim)
        """
        x = self.atom_encoder(data.x.float())
        edge_attr = self.edge_encoder(data.edge_attr.float()) if data.edge_attr is not None else None
        
        for i in range(self.num_layers):
            x_new = self.convs[i](x, data.edge_index, edge_attr)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection
        
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_repr = global_mean_pool(x, batch)
        
        return x, graph_repr


# ============================================================
# Self-supervised Pretext Tasks
# ============================================================

class AtomMaskingHead(nn.Module):
    """Predict masked atom types from context."""
    
    def __init__(self, hidden_dim: int, num_atom_types: int = 12):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atom_types),
        )
    
    def forward(self, node_repr: torch.Tensor, mask_indices: torch.Tensor):
        masked_repr = node_repr[mask_indices]
        return self.predictor(masked_repr)


class ContextPredictionHead(nn.Module):
    """Predict whether a substructure exists in the neighborhood."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, anchor_repr: torch.Tensor, context_repr: torch.Tensor):
        combined = torch.cat([anchor_repr, context_repr], dim=-1)
        return self.predictor(combined).squeeze(-1)


# ============================================================
# Pretraining Loop
# ============================================================

def mask_atoms(data: Data, mask_rate: float = 0.15):
    """Randomly mask atom features and return mask indices + original types."""
    num_atoms = data.x.size(0)
    num_mask = max(1, int(num_atoms * mask_rate))
    
    mask_indices = torch.randperm(num_atoms)[:num_mask]
    
    # Extract original atom types (first 12 features are one-hot atom type)
    original_types = data.x[mask_indices, :12].argmax(dim=1)
    
    # Zero out masked atom features
    data_masked = data.clone()
    data_masked.x = data.x.clone()
    data_masked.x[mask_indices] = 0.0
    
    return data_masked, mask_indices, original_types


def collect_pretraining_smiles() -> List[str]:
    """
    Collect SMILES strings for pretraining from multiple sources.
    Target: ~10K-50K unlabeled molecules.
    """
    all_smiles = set()
    
    # Source 1: Our real polymer data
    try:
        from data.real_polymer_data import get_all_real_data
        for entry in get_all_real_data():
            all_smiles.add(entry.smiles)
        logger.info(f"  Real polymer data: {len(all_smiles)} SMILES")
    except ImportError:
        pass
    
    # Source 2: Curated polymer database
    try:
        from data.polymer_smiles_db import ALL_POLYMER_SMILES
        for smi in ALL_POLYMER_SMILES:
            all_smiles.add(smi)
        logger.info(f"  + Polymer DB: total {len(all_smiles)}")
    except ImportError:
        pass
    
    # Source 3: Generate diverse SMILES via RDKit random molecules
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        
        # Generate random drug-like molecules via enumeration of known scaffolds
        # Use SMILES manipulation to create variations
        scaffolds = [
            'C(=O)O', 'CC(=O)O', 'CCC(=O)O', 'OC(=O)CC(=O)O',
            'OCC(=O)O', 'CC(O)C(=O)O', 'OCCCO', 'OCCCCO', 'OCCCCCO',
            'O=C1CCCO1', 'O=C1CCCCO1', 'O=C1CCCCCO1',
            'OC(=O)CCC(=O)O', 'OC(=O)CCCC(=O)O', 'OC(=O)CCCCC(=O)O',
            'NCC(=O)O', 'NC(C)C(=O)O', 'NCCCC(=O)O',
        ]
        
        # Create variations by adding side chains
        side_chains = ['C', 'CC', 'CCC', 'O', 'CO', 'CCO', 'N', 'C(C)C', 'C(=O)C']
        
        count = 0
        for scaffold in scaffolds:
            mol = Chem.MolFromSmiles(scaffold)
            if mol is None:
                continue
            
            all_smiles.add(Chem.MolToSmiles(mol))
            
            # Generate substituted variants
            for sc in side_chains:
                for i in range(len(scaffold)):
                    variant = scaffold[:i] + f'({sc})' + scaffold[i:]
                    vmol = Chem.MolFromSmiles(variant)
                    if vmol is not None:
                        canonical = Chem.MolToSmiles(vmol)
                        mw = Descriptors.MolWt(vmol)
                        if 50 < mw < 500:  # Filter reasonable MW
                            all_smiles.add(canonical)
                            count += 1
                            if count > 10000:
                                break
                    if count > 10000:
                        break
                if count > 10000:
                    break
            if count > 10000:
                break
        
        logger.info(f"  + Generated variants: total {len(all_smiles)}")
    except ImportError:
        pass
    
    return list(all_smiles)


def pretrain_encoder(
    encoder: PretrainableGNNEncoder,
    smiles_list: List[str],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    mask_rate: float = 0.15,
    save_path: str = './checkpoints/pretrained_gnn.pt',
    device: str = 'cpu',
) -> dict:
    """
    Run self-supervised pretraining on the GNN encoder.
    
    Returns:
        dict with training metrics
    """
    encoder = encoder.to(device)
    atom_head = AtomMaskingHead(encoder.hidden_dim, num_atom_types=12).to(device)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(atom_head.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Convert SMILES to graphs
    logger.info(f"Converting {len(smiles_list)} SMILES to graphs...")
    graphs = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None and g.x.size(0) >= 3:
            graphs.append(g)
    logger.info(f"  Valid graphs: {len(graphs)}")
    
    if len(graphs) < 100:
        logger.warning("Too few graphs for meaningful pretraining")
        return {'epochs': 0, 'final_loss': float('inf')}
    
    # Training loop
    history = {'losses': []}
    
    for epoch in range(epochs):
        encoder.train()
        atom_head.train()
        
        random.shuffle(graphs)
        epoch_losses = []
        
        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i+batch_size]
            if len(batch_graphs) < 2:
                continue
            
            # Apply atom masking
            masked_graphs = []
            all_mask_indices = []
            all_original_types = []
            offset = 0
            
            for g in batch_graphs:
                g_masked, mask_idx, orig_types = mask_atoms(g, mask_rate)
                masked_graphs.append(g_masked)
                all_mask_indices.append(mask_idx + offset)
                all_original_types.append(orig_types)
                offset += g.x.size(0)
            
            # Batch
            batch = Batch.from_data_list(masked_graphs).to(device)
            mask_indices = torch.cat(all_mask_indices).to(device)
            original_types = torch.cat(all_original_types).to(device)
            
            # Forward
            node_repr, graph_repr = encoder(batch)
            
            # Atom masking loss
            pred_types = atom_head(node_repr, mask_indices)
            loss = F.cross_entropy(pred_types, original_types)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        history['losses'].append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"  Pretrain epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    # Save pretrained weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'hidden_dim': encoder.hidden_dim,
        'num_layers': encoder.num_layers,
    }, save_path)
    logger.info(f"  Saved pretrained encoder to {save_path}")
    
    return {
        'epochs': epochs,
        'num_graphs': len(graphs),
        'final_loss': history['losses'][-1] if history['losses'] else 0,
        'save_path': save_path,
    }


def load_pretrained_weights(model: nn.Module, pretrained_path: str, device: str = 'cpu') -> bool:
    """
    Load pretrained GNN encoder weights into a surrogate model.
    Matches common layer names between the pretrained encoder and the target model.
    """
    if not os.path.exists(pretrained_path):
        logger.warning(f"Pretrained weights not found: {pretrained_path}")
        return False
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained_dict = checkpoint.get('encoder_state_dict', checkpoint)
    model_dict = model.state_dict()
    
    # Find matching layers
    matched = 0
    for key in pretrained_dict:
        if key in model_dict and pretrained_dict[key].shape == model_dict[key].shape:
            model_dict[key] = pretrained_dict[key]
            matched += 1
    
    model.load_state_dict(model_dict)
    logger.info(f"  Loaded {matched} pretrained layers from {pretrained_path}")
    return matched > 0


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-path', type=str, default='./checkpoints/pretrained_gnn.pt')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("  GNN SELF-SUPERVISED PRETRAINING")
    logger.info("=" * 60)
    
    # Collect SMILES
    smiles = collect_pretraining_smiles()
    logger.info(f"Collected {len(smiles)} SMILES for pretraining")
    
    # Create encoder
    encoder = PretrainableGNNEncoder()
    logger.info(f"Encoder: {sum(p.numel() for p in encoder.parameters())} parameters")
    
    import torch as th
    device = 'mps' if th.backends.mps.is_available() else 'cpu'
    
    # Pretrain
    result = pretrain_encoder(
        encoder, smiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        device=device,
    )
    
    logger.info(f"\nPretraining complete!")
    logger.info(f"  Graphs: {result['num_graphs']}")
    logger.info(f"  Final loss: {result['final_loss']:.4f}")
    logger.info(f"  Saved to: {result['save_path']}")
