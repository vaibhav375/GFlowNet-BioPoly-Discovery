"""
GFlowNet Policy Network
========================
GNN-based policy network that maps partial molecular graphs to 
action probability distributions for the GFlowNet.

The policy network answers: "Given the current partial molecule,
what atom should I add next, or what bond should I form?"

Actions:
    1. Add atom (C, N, O, S, F, Cl, Br, P) to a specific position
    2. Form bond (single, double, triple) between existing atoms
    3. Terminate (molecule is complete)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
    global_add_pool,
    BatchNorm,
)
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional, List


# Action space definition
ATOM_ACTIONS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P']  # 8 atom types

# Fragment actions — pre-built molecular fragments for critical functional groups
# These allow the GFlowNet to construct ester, amide, and carbonyl bonds in a single
# step, rather than the multi-step atom+double-bond path that was never discovered.
# Inspired by RGFN (Koziarski et al., NeurIPS 2024) fragment-based GFlowNets.
FRAGMENT_ACTIONS = [
    'C(=O)O',    # Ester linkage fragment (THE key for biodegradation)
    'C(=O)N',    # Amide linkage fragment (peptide bonds, nylons)
    'C(=O)',     # Carbonyl (ketone/aldehyde)
    'C(=O)OH',   # Carboxylic acid (polyester monomer end)
    'OC(=O)O',   # Carbonate linkage
    'C=C',       # Vinyl/double bond (addition polymerization)
]

BOND_ACTIONS = ['SINGLE', 'DOUBLE', 'TRIPLE']                  # 3 bond types
STOP_ACTION = 1                                                 # 1 stop action

NUM_ATOM_ACTIONS = len(ATOM_ACTIONS)        # 8
NUM_FRAGMENT_ACTIONS = len(FRAGMENT_ACTIONS)  # 6
NUM_BOND_ACTIONS = len(BOND_ACTIONS)        # 3
TOTAL_ACTIONS = NUM_ATOM_ACTIONS + NUM_FRAGMENT_ACTIONS + NUM_BOND_ACTIONS + STOP_ACTION  # 18


class GINEBlock(nn.Module):
    """Graph Isomorphism Network with Edge features block."""
    
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        
        nn_module = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        
        self.conv = GINEConv(nn_module, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = F.relu(h)
        return h


class PolicyNetwork(nn.Module):
    """
    GFlowNet Forward Policy P_F(a | s).
    
    Maps state (partial molecular graph) to action probabilities.
    
    Architecture:
        1. Node embedding (atom features → hidden_dim)
        2. GINE convolution layers with skip connections
        3. Global graph pooling
        4. Action head: MLP → softmax over action space
        5. Log Z head: Estimates partition function (for TB loss)
    
    Args:
        atom_feature_dim: Dimension of atom features
        bond_feature_dim: Dimension of bond features  
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 38,
        bond_feature_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(bond_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GINEBlock(hidden_dim, hidden_dim, hidden_dim))
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Graph-level representation (mean + sum pooling)
        graph_dim = 2 * hidden_dim
        
        # Forward policy head (P_F)
        self.forward_policy = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, TOTAL_ACTIONS),
        )
        
        # Backward policy head (P_B) — for trajectory balance
        self.backward_policy = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, TOTAL_ACTIONS),
        )
        
        # Log partition function estimator (log Z)
        self.log_z = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # State value function (for flow matching)
        self.state_flow = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def encode_graph(self, data: Data) -> torch.Tensor:
        """
        Encode a molecular graph into a fixed-size vector.
        
        Returns:
            Graph embedding [batch_size, 2 * hidden_dim]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode node and edge features
        h = self.node_encoder(x)
        
        # Only run GNN message-passing if there are edges.
        # Single-atom / edgeless graphs cause MPS backend errors
        # ("Placeholder tensor is empty!") with empty edge tensors.
        has_edges = edge_index is not None and edge_index.shape[1] > 0
        
        if has_edges:
            if edge_attr is not None and edge_attr.shape[0] > 0:
                e = self.edge_encoder(edge_attr)
            else:
                e = torch.zeros(edge_index.shape[1], self.hidden_dim, device=h.device)
            
            # GNN with skip connections
            for i in range(self.num_layers):
                h_prev = h
                h = self.gnn_layers[i](h, edge_index, e)
                h = self.dropout(h)
                h = self.layer_norms[i](h + h_prev)
        else:
            # No edges — just apply layer norms so shape is consistent
            for i in range(self.num_layers):
                h = self.layer_norms[i](h)
        
        # Pooling
        h_mean = global_mean_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        
        return torch.cat([h_mean, h_sum], dim=-1)
    
    def get_forward_policy(
        self,
        data: Data,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward policy P_F(a | s).
        
        Args:
            data: Current state (partial molecule graph)
            action_mask: Boolean mask of valid actions [batch_size, TOTAL_ACTIONS]
            temperature: Sampling temperature (lower = more greedy)
        
        Returns:
            log_probs: Log probabilities of each action [batch_size, TOTAL_ACTIONS]
            probs: Action probabilities [batch_size, TOTAL_ACTIONS]
        """
        graph_embed = self.encode_graph(data)
        logits = self.forward_policy(graph_embed)
        
        # Apply temperature
        logits = logits / max(temperature, 1e-8)
        
        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        return log_probs, probs
    
    def get_backward_policy(
        self,
        data: Data,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute backward policy P_B(a | s).
        Used in trajectory balance loss.
        """
        graph_embed = self.encode_graph(data)
        logits = self.backward_policy(graph_embed)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    def get_log_z(self, data: Data) -> torch.Tensor:
        """Estimate log partition function log Z."""
        graph_embed = self.encode_graph(data)
        return self.log_z(graph_embed).squeeze(-1)
    
    def get_state_flow(self, data: Data) -> torch.Tensor:
        """Estimate state flow F(s)."""
        graph_embed = self.encode_graph(data)
        return self.state_flow(graph_embed).squeeze(-1)
    
    def sample_action(
        self,
        data: Data,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> Tuple[int, torch.Tensor]:
        """
        Sample an action from the forward policy.
        
        Args:
            data: Current state
            action_mask: Valid action mask
            temperature: Sampling temperature
            epsilon: Random action probability (for exploration)
        
        Returns:
            action_idx: Sampled action index
            log_prob: Log probability tensor (retains gradient for TB loss!)
        """
        log_probs, probs = self.get_forward_policy(data, action_mask, temperature)
        
        # Epsilon-greedy exploration
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            # Random valid action
            if action_mask is not None:
                valid_actions = action_mask[0].nonzero(as_tuple=True)[0]
                action_idx = valid_actions[torch.randint(len(valid_actions), (1,))].item()
            else:
                action_idx = torch.randint(TOTAL_ACTIONS, (1,)).item()
            log_prob = log_probs[0, action_idx]  # Keep as tensor for gradient
        else:
            # Sample from policy
            action_idx = torch.multinomial(probs[0], 1).item()
            log_prob = log_probs[0, action_idx]  # Keep as tensor for gradient
        
        return action_idx, log_prob


def create_policy_network(config: dict = None) -> PolicyNetwork:
    """Factory function to create policy network."""
    if config is None:
        config = {}
    
    return PolicyNetwork(
        atom_feature_dim=config.get('atom_feature_dim', 38),
        bond_feature_dim=config.get('bond_feature_dim', 7),
        hidden_dim=config.get('policy_hidden_dim', 256),
        num_layers=config.get('policy_num_layers', 5),
        dropout=config.get('policy_dropout', 0.1),
    )
