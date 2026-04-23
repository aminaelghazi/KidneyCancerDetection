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
