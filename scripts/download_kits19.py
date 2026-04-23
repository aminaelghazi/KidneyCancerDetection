import subprocess
import os

def download_kits19(output_dir="data/kits19"):
    os.makedirs(output_dir, exist_ok=True)
    # Utilise l'outil officiel kits19 (basé sur git lfs)
    subprocess.run(["git", "clone", "https://github.com/neheller/kits19"], cwd=output_dir)
    print("KiTS19 téléchargé. Puis exécutez python -m kits19.download dans le dossier.")
