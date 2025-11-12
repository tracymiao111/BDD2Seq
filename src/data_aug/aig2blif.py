#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil
from tqdm import tqdm  # For progress bar

# Configuration
INPUT_AIG_DIR = ""
OUTPUT_BLIF_DIR = ""
AIGTOBLIF_PATH = ""  # Path to aigtoblif executable

# Options for aigtoblif (modify as needed)
AIGTOBLIF_OPTIONS = []  # e.g., ['-p', 'prefix_'] if needed

def ensure_directories():
    """
    Ensures that the input and output directories exist.
    """
    if not os.path.isdir(INPUT_AIG_DIR):
        print(f"Error: Input directory '{INPUT_AIG_DIR}' does not exist.")
        sys.exit(1)
    
    os.makedirs(OUTPUT_BLIF_DIR, exist_ok=True)

def ensure_aigtoblif():
    """
    Ensures that the aigtoblif executable exists and is executable.
    """
    if not os.path.isfile(AIGTOBLIF_PATH):
        print(f"Error: aigtoblif executable not found at '{AIGTOBLIF_PATH}'. Please verify the path.")
        sys.exit(1)
    
    if not os.access(AIGTOBLIF_PATH, os.X_OK):
        print(f"Error: aigtoblif at '{AIGTOBLIF_PATH}' is not executable. Setting executable permissions.")
        try:
            os.chmod(AIGTOBLIF_PATH, 0o755)
        except Exception as e:
            print(f"Failed to set executable permissions: {e}")
            sys.exit(1)

def convert_aig_to_blif(aig_file, blif_file):
    """
    Converts a single AIG file to BLIF format using aigtoblif.
    """
    try:
        command = [AIGTOBLIF_PATH] + AIGTOBLIF_OPTIONS + [aig_file, blif_file]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Error converting '{aig_file}' to BLIF:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception occurred while converting '{aig_file}': {e}")
        return False

def main():
    ensure_directories()
    ensure_aigtoblif()
    
    # List all AIG files in the input directory
    aig_files = [f for f in os.listdir(INPUT_AIG_DIR) if f.endswith('.aig') or f.endswith('.aag')]
    total_files = len(aig_files)
    
    if total_files == 0:
        print(f"No AIG files found in '{INPUT_AIG_DIR}'.")
        sys.exit(0)
    
    print(f"Starting conversion of {total_files} AIG files to BLIF format...")
    
    # Process each AIG file with a progress bar
    for aig_file in tqdm(aig_files, desc="Converting AIG to BLIF"):
        aig_path = os.path.join(INPUT_AIG_DIR, aig_file)
        blif_filename = os.path.splitext(aig_file)[0] + '.blif'
        blif_path = os.path.join(OUTPUT_BLIF_DIR, blif_filename)
        
        success = convert_aig_to_blif(aig_path, blif_path)
        if not success:
            print(f"Failed to convert '{aig_file}'. Check the error message above.")
    
    print(f"\nConversion complete. BLIF files are saved in '{OUTPUT_BLIF_DIR}'.")

if __name__ == "__main__":
    main()