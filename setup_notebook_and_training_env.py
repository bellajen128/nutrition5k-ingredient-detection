#!/usr/bin/env python3
"""
Setup script for Nutrition5K project on Northeastern HPC (Discovery)

This version:
1. Builds a dual environment setup (Notebook + Training venv)
2. Checks for local Nutrition5K dataset in /scratch/$USER/nutrition5k_cache
3. Skips any external Kaggle download (because HPC cannot connect outside)
4. Verifies and extracts manual uploads if present

Author: Bella Jen
Date: 2025-10-30
"""

import os
import sys
import subprocess
import getpass
import zipfile

# =========================================================
# Step 0: Define shared paths
# =========================================================
USER = getpass.getuser()
home_dir = os.path.expanduser("~")
scratch_dir = f"/scratch/{USER}"
cache_dir = f"{scratch_dir}/nutrition5k_cache"
venv_dir = f"{home_dir}/nutrition5k_env"

dataset_dir = f"{cache_dir}/datasets/siddhantrout/nutrition5k-dataset"
manual_zip = f"{cache_dir}/manual_upload/archive.zip"

print(f"Home dir: {home_dir}")
print(f"Scratch dir: {scratch_dir}")
print(f"KaggleHub cache: {cache_dir}")
print(f"Virtual env: {venv_dir}\n")

# =========================================================
# Step 1: Prepare cache directory
# =========================================================
os.makedirs(cache_dir, exist_ok=True)
print("Created/verified shared cache directory.\n")

# =========================================================
# Step 2: Install minimal user packages for Notebook use
# =========================================================
print("Installing basic Notebook libraries (user mode)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--user", "-q", "pandas", "openpyxl"],
    check=True
)
print("Notebook environment ready.\n")

# =========================================================
# Step 3: Setup training virtual environment
# =========================================================
if not os.path.exists(venv_dir):
    print("Creating training virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
else:
    print("Training virtual environment already exists.")

print("Installing training libraries inside venv...")
subprocess.run(
    [f"{venv_dir}/bin/python", "-m", "pip", "install", "--upgrade", "pip"],
    check=True
)
subprocess.run(
    [f"{venv_dir}/bin/python", "-m", "pip", "install", "-q",
     "torch", "torchvision", "ultralytics", "pandas", "openpyxl"],
    check=True
)
print("Training environment setup complete.\n")

# =========================================================
# Step 4: Dataset verification
# =========================================================
print("Checking for Nutrition5K dataset...")

# Case 1: Already exists (e.g., from previous upload)
if os.path.exists(dataset_dir) and any(fname.endswith((".xlsx", ".pkl")) for fname in os.listdir(dataset_dir)):
    print(f"Dataset already found at: {dataset_dir}")

# Case 2: Manual zip found, extract it
elif os.path.exists(manual_zip):
    print(f"Found manual upload at: {manual_zip}")
    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(manual_zip, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    print(f"Dataset extracted to: {dataset_dir}")

# Case 3: Neither found → ask user to upload manually
else:
    print("\n*** Dataset not found! ***")
    print("Please manually upload the Nutrition5K archive.zip file to:")
    print(f"  {manual_zip}")
    print("\nYou can download it from:")
    print("  https://www.kaggle.com/datasets/siddhantrout/nutrition5k-dataset")
    print("Then re-run this script.\n")
    sys.exit(1)

# =========================================================
# Step 5: Quick data verification
# =========================================================
import pandas as pd
xlsx_files = [f for f in os.listdir(dataset_dir) if f.endswith(".xlsx")]
print("\nFiles in dataset folder:", xlsx_files)

if xlsx_files:
    sample = os.path.join(dataset_dir, xlsx_files[0])
    print(f"Preview of {os.path.basename(sample)}:")
    df = pd.read_excel(sample)
    print(df.head())
else:
    print("Warning: no .xlsx files detected. Verify dataset integrity manually.\n")

print("\nAll setup steps complete.")
print("Notebook analysis: use system Python (pip install --user)")
print(f"Model training:   activate with 'source {venv_dir}/bin/activate'\n")
