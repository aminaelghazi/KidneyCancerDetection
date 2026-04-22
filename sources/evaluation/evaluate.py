#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for kidney tumor classification model.
Computes standard metrics (accuracy, precision, recall, F1, AUC, confusion matrix),
generates ROC curve and saves results to disk.

Usage:
    python evaluate.py --config configs/train_config.yaml --model_path results/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import yaml
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from sources.models.resnet18 import ModifiedResNet18

# ---------- Configuration du logging ----------
def setup_logging(log_dir=None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(level=logging.INFO, format=log_format,
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)

# ---------- Dataset (identique à l'entraînement, sans augmentation) ----------
class KidneySliceDataset(torch.utils.data.Dataset):
    """Dataset for evaluation (no augmentation)."""
    def __init__(self, slices, labels):
        self.slices = slices.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.slices[idx]
        # Normalize to [-1, 1] (same as training)
        img = (img - 0.5) / 0.5   # assuming input already in [0,1] after preprocessing
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(self.labels[idx])
        return img, label

# ---------- Chargement des données ----------
def load_test_data(slices_path, labels_path):
    """Load test set numpy arrays."""
    slices = np.load(slices_path)
    labels = np.load(labels_path)
    logging.info(f"Loaded test data: {len(slices)} slices, class distribution: {np.bincount(labels)}")
    return slices, labels

# ---------- Évaluation ----------
def evaluate(model, dataloader, device, threshold=0.5):
    """Run evaluation and return predictions, probabilities, and metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ---------- Calcul des métriques ----------
def compute_metrics(labels, preds, probs):
    """Compute all classification metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    metrics['f1'] = f1_score(labels, preds, zero_division=0)
    try:
        metrics['auc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auc'] = 0.5
    metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    metrics['classification_report'] = classification_report(labels, preds, output_dict=True)
    return metrics

# ---------- Sauvegarde des figures ----------
def save_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")

def save_roc_curve(labels, probs, output_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"ROC curve saved to {output_path}")

# ---------- Fonction principale ----------
def main(config_path, model_path, output_dir=None):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(config['paths']['log_dir'], 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Config: {config_path}")
    logger.info(f"Model: {model_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load test data
    slices_path = config['data'].get('test_slices_path')
    labels_path = config['data'].get('test_labels_path')
    if not slices_path or not labels_path:
        # Fallback to default processed data
        base = os.path.dirname(config['data']['slices_path'])
        slices_path = os.path.join(base, 'test_slices.npy')
        labels_path = os.path.join(base, 'test_labels.npy')
        logger.warning(f"Test data paths not in config, using fallback: {slices_path}, {labels_path}")

    slices, labels = load_test_data(slices_path, labels_path)

    # Dataset and loader
    dataset = KidneySliceDataset(slices, labels)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                        shuffle=False, num_workers=config['training']['num_workers'])

    # Load model
    model = ModifiedResNet18(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info("Model loaded successfully")

    # Evaluate
    true_labels, preds, probs = evaluate(model, loader, device, threshold=0.5)

    # Compute metrics
    metrics = compute_metrics(true_labels, preds, probs)
    logger.info("Evaluation results:")
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'classification_report']:
            logger.info(f"  {key}: {value:.4f}")

    # Save metrics to JSON
    metrics_json_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_json_path}")

    # Save figures
    class_names = config['data'].get('class_names', ['normal', 'tumor'])
    cm = np.array(metrics['confusion_matrix'])
    save_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    save_roc_curve(true_labels, probs, os.path.join(output_dir, 'roc_curve.png'))

    logger.info("Evaluation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate kidney tumor classification model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    args = parser.parse_args()
    main(args.config, args.model_path, args.output_dir)
