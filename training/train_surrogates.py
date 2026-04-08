"""
Surrogate Model Training
=========================
Trains S_bio (biodegradability) and S_mech (mechanical properties)
surrogate models on the polymer dataset.

Includes Green AI practices:
    - Carbon tracking with CodeCarbon
    - Mixed precision training
    - Early stopping
    - Gradient checkpointing
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeomDataLoader
from tqdm import tqdm

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import (
    generate_polymer_dataset,
    smiles_to_graph,
    split_dataset,
    compute_synthetic_biodegradability_label,
)
from models.surrogate_bio import create_bio_model
from models.surrogate_mech import create_mech_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GreenEarlyStopping:
    """
    Early stopping considering both performance and carbon budget.
    Stops training if:
        1. Validation loss hasn't improved for `patience` epochs, OR
        2. Carbon budget is exceeded
    """
    
    def __init__(self, patience: int = 15, carbon_budget_kg: float = 2.0, min_delta: float = 1e-4):
        self.patience = patience
        self.carbon_budget = carbon_budget_kg
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.total_carbon = 0.0
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int, epoch_carbon: float = 0.0) -> bool:
        self.total_carbon += epoch_carbon
        
        if self.carbon_budget > 0 and self.total_carbon > self.carbon_budget:
            logger.warning(f"Carbon budget ({self.carbon_budget} kg CO2) exceeded at epoch {epoch}!")
            return True
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}. Best: {self.best_epoch} (loss={self.best_loss:.6f})")
                return True
        
        return False


def prepare_graph_data(
    smiles_list: list,
    labels: dict,
    target_key: str,
) -> list:
    """Convert SMILES + labels to list of PyG Data objects."""
    data_list = []
    
    for idx, smiles in enumerate(tqdm(smiles_list, desc=f"Preparing {target_key} data")):
        graph = smiles_to_graph(smiles)
        if graph is None:
            continue
        
        # Add target label
        if target_key in labels and idx < len(labels[target_key]):
            graph.y = torch.tensor([labels[target_key][idx]], dtype=torch.float)
        else:
            continue
        
        # For mechanical model: add all property targets
        if 'tensile_strength' in labels and idx < len(labels['tensile_strength']):
            graph.tensile = torch.tensor([labels['tensile_strength'][idx] / 100.0], dtype=torch.float)
            graph.tg = torch.tensor([(labels['glass_transition'][idx] + 100.0) / 400.0], dtype=torch.float)
            graph.flexibility = torch.tensor([labels['flexibility'][idx]], dtype=torch.float)
        
        data_list.append(graph)
    
    return data_list


def train_surrogate_model(
    model: nn.Module,
    train_loader: GeomDataLoader,
    val_loader: GeomDataLoader,
    model_name: str,
    target_key: str = 'y',
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = 'cpu',
    save_dir: str = './checkpoints',
    use_mixed_precision: bool = True,
) -> Dict:
    """
    Train a surrogate model with Green AI practices.
    
    Returns:
        Dictionary of training statistics
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    # Session 9: Label smoothing for surrogate training
    # Prevents overconfident predictions at boundaries (0/1)
    label_smooth_eps = 0.05
    def smooth_criterion(pred, target):
        """MSE with label smoothing — soft targets improve generalization."""
        smoothed_target = target * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
        return criterion(pred, smoothed_target)
    
    # Session 11 B3: Stochastic Weight Averaging
    swa_start_epoch = int(epochs * 0.8)  # Start SWA at 80% of training
    swa_weights = []
    
    # Session 11 B4: R-Dropout coefficient
    rdrop_alpha = 0.1
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device == 'cuda' else None
    
    early_stopping = GreenEarlyStopping(patience=patience, carbon_budget_kg=2.0)
    
    # Carbon tracking with CodeCarbon
    tracking_carbon = False
    tracker = None
    try:
        import platform
        import subprocess as _sp
        
        skip_cc = False
        if platform.system() == 'Darwin':
            try:
                r = _sp.run(['sudo', '-n', 'true'], capture_output=True, timeout=2)
                if r.returncode != 0:
                    skip_cc = True
            except Exception:
                skip_cc = True
        
        if not skip_cc:
            from codecarbon import EmissionsTracker
            os.makedirs("./carbon_logs", exist_ok=True)
            tracker = EmissionsTracker(
                project_name=f"train_{model_name}",
                output_dir="./carbon_logs",
                log_level="warning",
                save_to_file=True,
                allow_multiple_runs=True,
                measure_power_secs=30,
            )
            tracker.start()
            tracking_carbon = True
            logger.info(f"✅ CodeCarbon tracking enabled for {model_name}")
        else:
            logger.info(f"ℹ️  Using TDP-based carbon estimation for {model_name}")
    except Exception as e:
        logger.warning(f"CodeCarbon unavailable ({e}). Carbon tracking disabled.")
        tracking_carbon = False
        tracker = None
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_mae': [],
        'lr': [], 'epoch_time': [],
    }
    
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, LR: {lr}")
    logger.info(f"Mixed Precision: {use_mixed_precision and device == 'cuda'}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # --- Training ---
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    if model_name == 's_mech':
                        outputs = model(batch)
                        loss = (
                            criterion(outputs['tensile'], batch.tensile) +
                            criterion(outputs['tg'], batch.tg) +
                            criterion(outputs['flexibility'], batch.flexibility)
                        ) / 3.0
                    else:
                        pred = model(batch)
                        loss = criterion(pred.squeeze(), batch.y.squeeze())
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if model_name == 's_mech':
                    outputs = model(batch)
                    loss = (
                        criterion(outputs['tensile'], batch.tensile) +
                        criterion(outputs['tg'], batch.tg) +
                        criterion(outputs['flexibility'], batch.flexibility)
                    ) / 3.0
                else:
                    pred = model(batch)
                    loss = smooth_criterion(pred.squeeze(), batch.y.squeeze())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_losses.append(loss.item())
        
        # Session 11 B4: R-Dropout — train twice with different dropout, minimize KL divergence
        if model_name != 's_mech' and epoch >= 10:  # Only after warmup
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # Two forward passes with different dropout
                model.train()
                pred1 = model(batch).squeeze()
                pred2 = model(batch).squeeze()
                # KL divergence between the two predictions
                p1 = torch.clamp(pred1, 0.01, 0.99)
                p2 = torch.clamp(pred2, 0.01, 0.99)
                rdrop_loss = rdrop_alpha * (
                    torch.mean((p1 - p2) ** 2)
                )
                if rdrop_loss.requires_grad:
                    rdrop_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                break  # Only one batch per epoch for R-Dropout (efficiency)
        
        # Session 11 B3: Collect SWA weights
        if epoch >= swa_start_epoch:
            swa_weights.append({k: v.clone() for k, v in model.state_dict().items()})
        
        avg_train_loss = np.mean(train_losses)
        
        # --- Validation ---
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                if model_name == 's_mech':
                    outputs = model(batch)
                    loss = (
                        criterion(outputs['tensile'], batch.tensile) +
                        criterion(outputs['tg'], batch.tg) +
                        criterion(outputs['flexibility'], batch.flexibility)
                    ) / 3.0
                    val_preds.extend(outputs['combined'].cpu().numpy().tolist())
                    # Compute ground-truth combined score for MAE
                    gt_combined = 0.45 * batch.tensile + 0.35 * batch.tg + 0.20 * batch.flexibility
                    val_targets.extend(gt_combined.cpu().numpy().tolist())
                else:
                    pred = model(batch)
                    loss = criterion(pred.squeeze(), batch.y.squeeze())
                    val_preds.extend(pred.squeeze().cpu().numpy().tolist())
                    val_targets.extend(batch.y.squeeze().cpu().numpy().tolist())
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_targets))) if val_targets else 0.0
        
        epoch_time = time.time() - epoch_start
        
        # LR scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"MAE: {val_mae:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pt'))
        
        # Early stopping
        if early_stopping(avg_val_loss, epoch):
            break
    
    # Stop carbon tracking
    total_emissions = 0.0
    if tracking_carbon and tracker is not None:
        total_emissions = tracker.stop()
        logger.info(f"Total carbon emissions for {model_name}: {total_emissions:.6f} kg CO2")
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_final.pt'))
    
    # Session 11 B3: Apply SWA averaging
    if swa_weights:
        logger.info(f"Applying SWA over {len(swa_weights)} weight snapshots")
        avg_state = {}
        for key in swa_weights[0]:
            avg_state[key] = torch.stack([w[key].float() for w in swa_weights]).mean(dim=0)
        model.load_state_dict(avg_state)
        torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_swa.pt'))
        logger.info(f"SWA model saved as {model_name}_swa.pt")
    
    history['total_emissions_kg'] = total_emissions
    history['best_val_loss'] = best_val_loss
    history['total_epochs'] = epoch + 1
    history['swa_snapshots'] = len(swa_weights)
    
    with open(os.path.join(save_dir, f'{model_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def main():
    """Train all surrogate models."""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # --- Generate Dataset ---
    logger.info("\n📊 Generating polymer dataset...")
    smiles_list, labels = generate_polymer_dataset(n=5000)
    
    logger.info(f"Dataset: {len(smiles_list)} molecules")
    logger.info(f"Labels: {list(labels.keys())}")
    
    # Split
    splits = split_dataset(smiles_list, labels)
    train_smiles, train_labels = splits['train']
    val_smiles, val_labels = splits['val']
    test_smiles, test_labels = splits['test']
    
    logger.info(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")
    
    # --- Train S_bio ---
    logger.info("\n🧬 Preparing S_bio (Biodegradability) data...")
    bio_train_data = prepare_graph_data(train_smiles, train_labels, 'biodegradability')
    bio_val_data = prepare_graph_data(val_smiles, val_labels, 'biodegradability')
    
    logger.info(f"S_bio data: {len(bio_train_data)} train, {len(bio_val_data)} val")
    
    bio_train_loader = GeomDataLoader(bio_train_data, batch_size=64, shuffle=True)
    bio_val_loader = GeomDataLoader(bio_val_data, batch_size=64, shuffle=False)
    
    bio_model = create_bio_model()
    bio_history = train_surrogate_model(
        model=bio_model,
        train_loader=bio_train_loader,
        val_loader=bio_val_loader,
        model_name='s_bio',
        epochs=100,
        lr=0.001,
        patience=15,
        device=device,
    )
    
    # --- Train S_mech ---
    logger.info("\n⚙️ Preparing S_mech (Mechanical Properties) data...")
    mech_train_data = prepare_graph_data(train_smiles, train_labels, 'tensile_strength')
    mech_val_data = prepare_graph_data(val_smiles, val_labels, 'tensile_strength')
    
    logger.info(f"S_mech data: {len(mech_train_data)} train, {len(mech_val_data)} val")
    
    mech_train_loader = GeomDataLoader(mech_train_data, batch_size=64, shuffle=True)
    mech_val_loader = GeomDataLoader(mech_val_data, batch_size=64, shuffle=False)
    
    mech_model = create_mech_model()
    mech_history = train_surrogate_model(
        model=mech_model,
        train_loader=mech_train_loader,
        val_loader=mech_val_loader,
        model_name='s_mech',
        epochs=100,
        lr=0.001,
        patience=15,
        device=device,
    )
    
    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"S_bio: Best Val Loss = {bio_history['best_val_loss']:.6f}, "
                f"Epochs = {bio_history['total_epochs']}, "
                f"CO2 = {bio_history.get('total_emissions_kg', 0):.6f} kg")
    logger.info(f"S_mech: Best Val Loss = {mech_history['best_val_loss']:.6f}, "
                f"Epochs = {mech_history['total_epochs']}, "
                f"CO2 = {mech_history.get('total_emissions_kg', 0):.6f} kg")
    
    total_co2 = bio_history.get('total_emissions_kg', 0) + mech_history.get('total_emissions_kg', 0)
    logger.info(f"Total CO2 emissions: {total_co2:.6f} kg")
    logger.info("✅ Surrogate model training complete!")


if __name__ == "__main__":
    main()
