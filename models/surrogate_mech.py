"""
Surrogate Model: S_mech — Mechanical Properties Predictor
==========================================================
Multi-output Graph Neural Network that predicts three mechanical
properties from molecular graph structure:

    1. Tensile Strength (MPa) — Force resistance before breaking
    2. Glass Transition Temperature Tg (°C) — Rigid/flexible transition
    3. Flexibility (0-1) — Elasticity measure

Architecture: Shared GNN backbone + multi-head regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    BatchNorm,
    GraphNorm,
)
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple


class AttentiveMPNNLayer(MessagePassing):
    """
    Attentive Message Passing layer.
    Uses attention to weigh neighbor contributions differently,
    which helps capture varying bond importance for mechanical properties.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, heads: int = 4):
        super().__init__(aggr='add')
        
        self.heads = heads
        self.head_dim = hidden_dim // heads
        
        # Attention computation
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, heads),
        )
        
        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        
        self.norm = GraphNorm(node_dim)
    
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Combine with original features
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        out = self.norm(out)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        # Compute attention weights
        cat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        attn = self.attn_mlp(cat)
        attn = F.softmax(attn, dim=0)
        attn = attn.mean(dim=-1, keepdim=True)  # Average over heads
        
        # Compute message
        msg = self.message_mlp(cat)
        
        return attn * msg


class MechanicalPropertiesPredictor(nn.Module):
    """
    S_mech: Predicts mechanical properties from molecular graph.
    
    Multi-task architecture with shared GNN backbone and 
    property-specific regression heads.
    
    Outputs:
        - tensile_strength: Predicted tensile strength (MPa), normalized [0, 1]
        - glass_transition: Predicted Tg (°C), normalized [0, 1]
        - flexibility: Predicted flexibility score [0, 1]
        - combined_score: Weighted combination for reward
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 38,
        bond_feature_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        num_properties: int = 3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_properties = num_properties
        
        # Property normalization ranges (for denormalization)
        self.register_buffer('ts_min', torch.tensor(0.0))
        self.register_buffer('ts_max', torch.tensor(100.0))
        self.register_buffer('tg_min', torch.tensor(-100.0))
        self.register_buffer('tg_max', torch.tensor(300.0))
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(bond_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Shared GNN backbone
        self.gnn_layers = nn.ModuleList([
            AttentiveMPNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Layer norms for skip connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Global pooling (mean + max + add = 3 * hidden_dim)
        pooling_dim = 3 * hidden_dim
        
        # Property-specific heads
        self.tensile_head = nn.Sequential(
            nn.Linear(pooling_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        self.tg_head = nn.Sequential(
            nn.Linear(pooling_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.flexibility_head = nn.Sequential(
            nn.Linear(pooling_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Dictionary with keys: 'tensile', 'tg', 'flexibility', 'combined'
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode
        h = self.node_encoder(x)
        if edge_attr is not None and edge_attr.shape[0] > 0:
            e = self.edge_encoder(edge_attr)
        else:
            e = torch.zeros(edge_index.shape[1], h.shape[-1], device=h.device)
        
        # GNN layers with skip connections
        for i in range(self.num_layers):
            h_prev = h
            h = self.gnn_layers[i](h, edge_index, e)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layer_norms[i](h + h_prev)  # Residual + LayerNorm
        
        # Multi-pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_add = global_add_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max, h_add], dim=-1)
        
        # Property predictions (squeeze to remove trailing dim)
        tensile = self.tensile_head(h_graph).squeeze(-1)
        tg = self.tg_head(h_graph).squeeze(-1)
        flexibility = self.flexibility_head(h_graph).squeeze(-1)
        
        # Combined mechanical score
        # Weighted combination rather than geometric mean, because
        # geometric mean is too harsh: a single low sub-property
        # collapses the entire score.
        # Weights: tensile 0.45, tg 0.35, flexibility 0.20
        combined = 0.45 * tensile + 0.35 * tg + 0.20 * flexibility
        
        return {
            'tensile': tensile,
            'tg': tg,
            'flexibility': flexibility,
            'combined': combined,
        }
    
    def predict_score(self, data: Data) -> float:
        """Get combined mechanical score for reward computation."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
        return outputs['combined'].item()
    
    def predict_with_uncertainty(
        self, data: Data, num_samples: int = 10
    ) -> Tuple[float, float]:
        """
        MC Dropout uncertainty estimation — Gal & Ghahramani (ICML 2016).
        
        Returns (mean_combined_score, uncertainty_std) from K stochastic
        forward passes with dropout enabled at inference time.
        """
        self.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.forward(data)
                predictions.append(outputs['combined'].item())
        self.eval()
        
        mean_pred = sum(predictions) / len(predictions)
        if len(predictions) > 1:
            variance = sum((p - mean_pred) ** 2 for p in predictions) / (len(predictions) - 1)
            uncertainty = variance ** 0.5
        else:
            uncertainty = 0.0
        
        return mean_pred, uncertainty
    
    def denormalize(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert normalized predictions back to physical units."""
        return {
            'tensile_mpa': outputs['tensile'] * (self.ts_max - self.ts_min) + self.ts_min,
            'tg_celsius': outputs['tg'] * (self.tg_max - self.tg_min) + self.tg_min,
            'flexibility': outputs['flexibility'],
        }


def create_mech_model(config: dict = None) -> MechanicalPropertiesPredictor:
    """Factory function to create S_mech model."""
    if config is None:
        config = {}
    
    return MechanicalPropertiesPredictor(
        atom_feature_dim=config.get('atom_feature_dim', 38),
        bond_feature_dim=config.get('bond_feature_dim', 7),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.2),
    )


if __name__ == "__main__":
    from data.preprocessing import smiles_to_graph
    
    model = create_mech_model()
    print(f"S_mech Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test
    graph = smiles_to_graph('CC(O)C(=O)O')
    if graph is not None:
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        outputs = model(graph)
        print(f"Tensile: {outputs['tensile'].item():.4f}")
        print(f"Tg: {outputs['tg'].item():.4f}")
        print(f"Flexibility: {outputs['flexibility'].item():.4f}")
        print(f"Combined: {outputs['combined'].item():.4f}")
    
    print("✓ S_mech model ready!")
