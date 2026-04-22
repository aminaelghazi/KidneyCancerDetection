#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a ResNet-18 model for kidney tumor classification from CT slices.
Supports patient-wise cross-validation, augmentation, checkpointing, and full metrics.
"""

import os
import sys
import random
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import SimpleITK as sitk
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from sources.models.resnet18 import ModifiedResNet18

# Set up logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom dataset with augmentation
class KidneySliceDataset(Dataset):
    """Dataset for kidney CT slices with optional augmentations."""
    def __init__(self, slices, labels, patient_ids=None, transform=None, augment=False):
        self.slices = slices  # numpy array (N, H, W)
        self.labels = labels  # numpy array (N,)
        self.patient_ids = patient_ids  # optional for grouping
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.slices[idx].astype(np.float32)
        label = self.labels[idx].astype(np.int64)

        # Convert to tensor and add channel dimension
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        if self.augment and self.transform:
            img = self.transform(img)

        return img, label, idx  # return index for potential debugging

# Augmentation transforms for training
def get_train_transforms():
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize to [-1, 1]
    ])

def get_val_transforms():
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# Load preprocessed data
def load_data(data_config):
    """Load preprocessed numpy arrays (slices, labels, patient_ids)."""
    slices_path = data_config.get('slices_path')
    labels_path = data_config.get('labels_path')
    patient_ids_path = data_config.get('patient_ids_path', None)

    if not os.path.exists(slices_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Data not found: {slices_path} or {labels_path}")

    slices = np.load(slices_path)
    labels = np.load(labels_path)
    patient_ids = np.load(patient_ids_path) if patient_ids_path and os.path.exists(patient_ids_path) else None

    logging.info(f"Loaded {len(slices)} slices, class distribution: {np.bincount(labels)}")
    return slices, labels, patient_ids

# Training one epoch
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds

# Validation
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1,
        'auc': epoch_auc
    }, all_labels, all_preds, all_probs

# Main training function with cross-validation
def train_fold(train_idx, val_idx, fold, config, device, logger, writer, data):
    slices, labels, patient_ids = data
    train_dataset = KidneySliceDataset(
        slices[train_idx], labels[train_idx],
        transform=get_train_transforms(),
        augment=True
    )
    val_dataset = KidneySliceDataset(
        slices[val_idx], labels[val_idx],
        transform=get_val_transforms(),
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    model = ModifiedResNet18(num_classes=2, pretrained=config['model']['pretrained']).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config['training']['class_weights']).to(device) if config['training'].get('class_weights') else None)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config['training']['patience'], verbose=True)

    # For mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_metrics = {'acc': 0.0, 'f1': 0.0}
    best_epoch = -1
    early_stop_counter = 0
    early_stop_patience = config['training']['early_stop_patience']

    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"Fold {fold} - Epoch {epoch}/{config['training']['epochs']}")

        train_loss, train_acc, _, _ = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics, _, _, _ = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_metrics['loss'])

        # Log metrics
        writer.add_scalar(f'Fold_{fold}/train_loss', train_loss, epoch)
        writer.add_scalar(f'Fold_{fold}/train_acc', train_acc, epoch)
        writer.add_scalar(f'Fold_{fold}/val_loss', val_metrics['loss'], epoch)
        writer.add_scalar(f'Fold_{fold}/val_acc', val_metrics['acc'], epoch)
        writer.add_scalar(f'Fold_{fold}/val_auc', val_metrics['auc'], epoch)

        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        # Save checkpoint if best
        if val_metrics['acc'] > best_val_metrics['acc']:
            best_val_metrics = val_metrics
            best_epoch = epoch
            early_stop_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'fold': fold
            }
            torch.save(checkpoint, os.path.join(config['paths']['checkpoint_dir'], f'best_model_fold{fold}.pth'))
            logger.info(f"  *** New best model saved (Acc: {val_metrics['acc']:.4f}) ***")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping after {epoch} epochs")
                break

    logger.info(f"Fold {fold} completed. Best val acc: {best_val_metrics['acc']:.4f} at epoch {best_epoch}")
    return best_val_metrics

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(config['paths']['tensorboard_dir'], exist_ok=True)

    logger = setup_logging(config['paths']['log_dir'])
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Set seed
    set_seed(config['training']['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    slices, labels, patient_ids = load_data(config['data'])
    # For patient-wise cross-validation, we need patient_ids. If not available, fallback to random split.
    if patient_ids is not None:
        n_patients = len(np.unique(patient_ids))
        logger.info(f"Patient-wise cross-validation: {n_patients} patients")
        # GroupKFold
        group_kfold = GroupKFold(n_splits=config['training']['n_folds'])
        splits = list(group_kfold.split(slices, labels, groups=patient_ids))
    else:
        logger.warning("No patient IDs provided. Using random stratified split (may leak slices from same patient across folds).")
        skf = StratifiedKFold(n_splits=config['training']['n_folds'], shuffle=True, random_state=config['training']['seed'])
        splits = list(skf.split(slices, labels))

    # TensorBoard writer
    writer = SummaryWriter(log_dir=config['paths']['tensorboard_dir'])

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'='*50}\nStarting Fold {fold+1}/{config['training']['n_folds']}\n{'='*50}")
        best_metrics = train_fold(train_idx, val_idx, fold+1, config, device, logger, writer, (slices, labels, patient_ids))
        fold_results.append(best_metrics)

    # Summary
    logger.info("\n" + "="*50)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*50)
    metrics_names = ['acc', 'precision', 'recall', 'f1', 'auc']
    for metric in metrics_names:
        values = [res[metric] for res in fold_results]
        logger.info(f"{metric.upper()}: mean = {np.mean(values):.4f} ± {np.std(values):.4f}")

    writer.close()
    logger.info("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train kidney cancer detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
