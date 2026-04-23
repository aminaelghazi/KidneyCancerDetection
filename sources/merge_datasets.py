import numpy as np
import os

# Charger KiTS19 (déjà converti)
kits19_slices = np.load("data/processed/kits19_slices.npy")
kits19_labels = np.load("data/processed/kits19_labels.npy")

# Charger les nouvelles slices normales
normal_slices = np.load("data/processed/moroccan_normal_slices.npy")
normal_labels = np.load("data/processed/moroccan_normal_labels.npy")

# Fusionner
all_slices = np.concatenate([kits19_slices, normal_slices])
all_labels = np.concatenate([kits19_labels, normal_labels])

# Sauvegarder pour l'entraînement
np.save("data/processed/all_slices.npy", all_slices)
np.save("data/processed/all_labels.npy", all_labels)

print(f"Total slices: {len(all_slices)} (tumor: {np.sum(all_labels)}, normal: {len(all_labels)-np.sum(all_labels)})")
