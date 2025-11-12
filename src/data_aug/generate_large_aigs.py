import subprocess
import os
import random
import sys

# Configuration
NUM_AIGS_TO_GENERATE = 12000
OUTPUT_DIR = "all_generated_aigs"
AIGFUZZ_PATH = "./aigfuzz"  # Path to the aigfuzz executable

# Ensure the aigfuzz executable exists
if not os.path.isfile(AIGFUZZ_PATH):
    print(f"Error: {AIGFUZZ_PATH} not found. Please ensure aigfuzz is in the current directory.")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_aig(seed, output_file):
    """
    Generates an AIG using aigfuzz with the given seed and outputs to output_file.
    """
    command = [AIGFUZZ_PATH, "-l", "-v", "-c", "-o", output_file, str(seed)]
    try:
        # Run the command and capture output
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Optionally, you can process result.stdout or result.stderr if needed
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating AIG with seed {seed}: {e.stderr.decode().strip()}")
        return False

def main():
    for i in range(1, NUM_AIGS_TO_GENERATE + 1):
        seed = random.randint(1, 1_000_000_000)
        aig_filename = f"large_circuit_{i}.aig"
        aig_path = os.path.join(OUTPUT_DIR, aig_filename)

        # Check if the file already exists to prevent overwriting
        if os.path.exists(aig_path):
            print(f"File {aig_filename} already exists. Skipping.")
            continue

        print(f"Generating {aig_filename} with seed {seed}...")
        success = generate_aig(seed, aig_path)
        if success:
            print(f"Successfully generated {aig_filename}.")
        else:
            print(f"Failed to generate {aig_filename}.")

    print(f"Completed generating {NUM_AIGS_TO_GENERATE} AIGs in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()