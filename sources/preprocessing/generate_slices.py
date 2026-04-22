import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse
from skimage.transform import resize

def extract_kidney_slices(volume_path, mask_path, output_dir):
    img = sitk.ReadImage(volume_path)
    mask = sitk.ReadImage(mask_path)
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)
    
    # Windowing: level=50, width=350 HU
    img_array = np.clip(img_array, 50 - 350//2, 50 + 350//2)
    img_array = (img_array - (50 - 350//2)) / 350.0
    
    slices = []
    labels = []
    for z in range(img_array.shape[0]):
        kidney_mask = (mask_array[z] == 1) | (mask_array[z] == 2)
        if np.sum(kidney_mask) > 0:
            slice_img = img_array[z]
            slice_resized = resize(slice_img, (128, 128), preserve_range=True)
            slices.append(slice_resized)
            tumor_present = np.any(mask_array[z] == 2)
            labels.append(1 if tumor_present else 0)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "slices.npy"), np.array(slices))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))
    return len(slices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    for case in os.listdir(args.data_dir):
        case_path = os.path.join(args.data_dir, case)
        if not os.path.isdir(case_path):
            continue
        vol_file = os.path.join(case_path, "imaging.nii.gz")
        seg_file = os.path.join(case_path, "segmentation.nii.gz")
        if os.path.exists(vol_file) and os.path.exists(seg_file):
            print(f"Processing {case}")
            out_subdir = os.path.join(args.output_dir, case)
            extract_kidney_slices(vol_file, seg_file, out_subdir)
