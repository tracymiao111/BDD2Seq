#!/usr/bin/env python3

import subprocess
import os
import random
import sys
from tqdm import tqdm  # For progress bar

# Configuration
NUM_AIGS_TO_GENERATE = 1500  # Number of AIG files to generate
OUTPUT_DIR = "/Users/chromemono/Downloads/aiger/all_generated_aigs"
AIGFUZZ_PATH = "/Users/chromemono/Downloads/aiger/aigfuzz"  # Path to aigfuzz executable
AIGFUZZ_OPTIONS = ["-s", "-c"]  # Additional options for aigfuzz

# Ensure the aigfuzz executable exists
if not os.path.isfile(AIGFUZZ_PATH):
    print(f"Error: aigfuzz executable not found at '{AIGFUZZ_PATH}'. Please verify the path.")
    sys.exit(1)

# Ensure aigfuzz has executable permissions
if not os.access(AIGFUZZ_PATH, os.X_OK):
    print(f"Error: aigfuzz at '{AIGFUZZ_PATH}' is not executable. Setting executable permissions.")
    try:
        os.chmod(AIGFUZZ_PATH, 0o755)
    except Exception as e:
        print(f"Failed to set executable permissions for aigfuzz: {e}")
        sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_aig(seed, output_file):
    """
    Generates an AIG file using aigfuzz with the given seed.
    """
    command = [AIGFUZZ_PATH] + AIGFUZZ_OPTIONS + ["-o", output_file, str(seed)]
    try:
        # Run the aigfuzz command
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating AIG with seed {seed}: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"Unexpected error generating AIG with seed {seed}: {e}")
        return False

def main():
    print(f"Starting generation of {NUM_AIGS_TO_GENERATE} small AIG circuits...")
    
    for i in tqdm(range(1, NUM_AIGS_TO_GENERATE + 1), desc="Generating AIGs"):
        seed = random.randint(1, 1_000_000_000)  # Random seed for diversity
        aig_filename = f"small_circuit_{i}.aig"
        aig_path = os.path.join(OUTPUT_DIR, aig_filename)
        
        # Check if the file already exists to prevent overwriting
        if os.path.exists(aig_path):
            print(f"File {aig_filename} already exists. Skipping.")
            continue
        
        success = generate_aig(seed, aig_path)
        if success:
            print(f"Successfully generated {aig_filename}.")
        else:
            print(f"Failed to generate {aig_filename}.")
    
    print(f"\nCompleted generating AIGs in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()