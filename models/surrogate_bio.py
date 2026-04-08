"""
Surrogate Model: S_bio — Biodegradability Predictor
====================================================
Graph Neural Network that predicts biodegradability score [0, 1]
from molecular graph structure.

Architecture:
    - Message Passing Neural Network (MPNN)
    - Global mean + max pooling (concatenated)
    - 2-layer MLP regression head
    - Dropout for regularization
    - Skip connections between GNN layers

Key biodegradability indicators learned:
    - Ester bond presence and density
    - Oxygen-to-carbon ratio
    - Chain flexibility (rotatable bonds)
    - Absence of persistent functional groups (halogens, aromatic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_max_pool,
    BatchNorm,
)
from torch_geometric.data import Data
from typing import Optional, Tuple


class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network layer with edge features.
    
    Message: m_{ij} = MLP([h_i || h_j || e_{ij}])
    Update:  h_i' = GRU(h_i, Σ_j m_{ij})
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add')
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update function (GRU-based)
        self.gru = nn.GRUCell(hidden_dim, node_dim)
        
        # Batch normalization
        self.bn = BatchNorm(node_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Propagate messages
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # GRU update
        h = self.gru(aggr, x)
        h = self.bn(h)
        
        return h
    
    def message(self, x_i, x_j, edge_attr):
        # Concatenate source, target node features and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class BiodegradabilityPredictor(nn.Module):
    """
    S_bio: Predicts biodegradability score from molecular graph.
    
    Architecture:
        1. Node embedding (linear projection)
        2. K message passing layers with skip connections
        3. Global pooling (mean + max concatenated)
        4. MLP regression head → sigmoid → [0, 1]
    
    Args:
        atom_feature_dim: Input atom feature dimension (default: 38)
        bond_feature_dim: Input bond feature dimension (default: 7)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of MPNN layers (default: 4)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 38,
        bond_feature_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Edge feature projection
        self.edge_encoder = nn.Sequential(
            nn.Linear(bond_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Message passing layers
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Skip connection projections (if dimensions differ)
        self.skip_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Readout MLP (mean+max pooling → 2*hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Atom features [num_atoms, atom_feature_dim]
                - edge_index: Bond connectivity [2, num_edges]
                - edge_attr: Bond features [num_edges, bond_feature_dim]
                - batch: Batch assignment [num_atoms]
        
        Returns:
            Biodegradability score [batch_size, 1] in [0, 1]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode node and edge features
        h = self.node_encoder(x)
        
        if edge_attr is not None and edge_attr.shape[0] > 0:
            e = self.edge_encoder(edge_attr)
        else:
            e = torch.zeros(edge_index.shape[1], h.shape[-1], device=h.device)
        
        # Message passing with skip connections
        for i in range(self.num_layers):
            h_prev = h
            h = self.mpnn_layers[i](h, edge_index, e)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Skip connection
            h = h + self.skip_projections[i](h_prev)
        
        # Global pooling: concatenate mean and max
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)
        
        # Regression head with sigmoid activation
        out = self.readout(h_graph)
        out = torch.sigmoid(out)
        
        return out
    
    def predict(self, data: Data) -> float:
        """Predict biodegradability for a single molecule."""
        self.eval()
        with torch.no_grad():
            score = self.forward(data)
        return score.item()
    
    def predict_with_uncertainty(
        self, data: Data, num_samples: int = 10
    ) -> Tuple[float, float]:
        """
        MC Dropout uncertainty estimation — Gal & Ghahramani (ICML 2016).
        
        Performs K stochastic forward passes with dropout enabled at inference
        time. Returns mean prediction and epistemic uncertainty (std dev).
        
        Higher uncertainty → model is less confident → prioritize for active learning.
        
        Args:
            data: Input molecular graph
            num_samples: Number of MC Dropout forward passes (K)
        
        Returns:
            (mean_prediction, uncertainty_std)
        """
        self.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                score = self.forward(data)
                predictions.append(score.item())
        self.eval()  # Restore eval mode
        
        mean_pred = sum(predictions) / len(predictions)
        if len(predictions) > 1:
            variance = sum((p - mean_pred) ** 2 for p in predictions) / (len(predictions) - 1)
            uncertainty = variance ** 0.5
        else:
            uncertainty = 0.0
        
        return mean_pred, uncertainty


def create_bio_model(config: dict = None) -> BiodegradabilityPredictor:
    """Factory function to create S_bio model from config."""
    if config is None:
        config = {}
    
    return BiodegradabilityPredictor(
        atom_feature_dim=config.get('atom_feature_dim', 38),
        bond_feature_dim=config.get('bond_feature_dim', 7),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.2),
    )


if __name__ == "__main__":
    # Quick test
    from data.preprocessing import smiles_to_graph
    
    model = create_bio_model()
    print(f"S_bio Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with a biodegradable molecule (PLA monomer)
    graph = smiles_to_graph('CC(O)C(=O)O')
    if graph is not None:
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        score = model.predict(graph)
        print(f"PLA monomer biodegradability (untrained): {score:.4f}")
    
    # Test with a non-biodegradable molecule (polyethylene-like)
    graph2 = smiles_to_graph('CCCCCCCCCC')
    if graph2 is not None:
        graph2.batch = torch.zeros(graph2.x.size(0), dtype=torch.long)
        score2 = model.predict(graph2)
        print(f"PE-like biodegradability (untrained): {score2:.4f}")
    
    print("✓ S_bio model ready!")
