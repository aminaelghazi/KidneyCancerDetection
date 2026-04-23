## Data preparation

### KiTS19 (public)
1. Run `python scripts/download_kits19.py` to clone the official repository.
2. Follow the instructions to download the actual images (requires `git-lfs`).
3. Run `python sources/preprocessing/generate_slices.py --data_dir data/kits19 --output_dir data/processed/kits19`

### Moroccan clinical data (private)
Due to patient privacy, the Moroccan dataset is not publicly distributed.
However, the preprocessing pipeline is fully compatible with any CT volume in NIfTI or DICOM format.
Place your anonymized volumes in a folder (e.g., `/path/to/moroccan`) and adjust `configs/data_paths.yaml`.
Then run `python scripts/prepare_moroccan_template.py` (adapt it to your folder structure).

### Adding custom 2D slices (e.g., Moroccan normal kidneys)

If you have a collection of 2D slices (PNG, JPG, or NPY) organised by patient, you can import them:

1. Place your folder tree like: `custom_data/patient001/slice1.png`, etc.
2. Run the import script:
   ```bash
   python sources/preprocessing/import_2d_slices.py --input_dir custom_data --output_file data/processed/custom_slices.npy --label 0
