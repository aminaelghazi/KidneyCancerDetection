#!/usr/bin/env python3
"""
Import 2D kidney slices (already extracted) into the training format.
Assumes a folder structure: root/patient_id/slice_*.png (or .npy).
Generates a single .npy file with all slices and labels (all 0 for normal).
"""

import os
import numpy as np
import argparse
from skimage.transform import resize
from skimage.io import imread
from tqdm import tqdm

def load_slices_from_folder(folder_path, target_size=(128,128)):
    """Load all images from a folder, resize, return numpy array (N, H, W)."""
    slices = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png','.jpg','.npy'))])
    for f in files:
        filepath = os.path.join(folder_path, f)
        if f.endswith('.npy'):
            img = np.load(filepath)
        else:
            img = imread(filepath)
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = img.mean(axis=-1)
        # Resize
        img_resized = resize(img, target_size, preserve_range=True, anti_aliasing=True)
        slices.append(img_resized.astype(np.float32))
    return np.array(slices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root folder containing patient subfolders")
    parser.add_argument("--output_file", required=True, help="Output .npy file for slices")
    parser.add_argument("--label", type=int, default=0, help="Label for these slices (0=normal,1=tumor)")
    args = parser.parse_args()

    all_slices = []
    patient_folders = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    for patient in tqdm(patient_folders):
        patient_path = os.path.join(args.input_dir, patient)
        slices = load_slices_from_folder(patient_path)
        all_slices.extend(slices)

    all_slices = np.array(all_slices)
    labels = np.full(len(all_slices), args.label, dtype=np.int64)

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, all_slices)
    np.save(args.output_file.replace('_slices.npy', '_labels.npy'), labels)
    print(f"Saved {len(all_slices)} slices with label {args.label} to {args.output_file}")

if __name__ == "__main__":
    main()
