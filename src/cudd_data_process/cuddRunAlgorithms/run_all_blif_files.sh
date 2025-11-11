#!/bin/bash

# Check if the folder containing BLIF files is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_with_blif_files>"
    exit 1
fi

# Input directory containing BLIF files provided by the user
BLIF_FOLDER="$1"

# Corrected list of reordering algorithms with corresponding integers
ALGORITHMS=(
  "CUDD_REORDER_RANDOM 2"
  "CUDD_REORDER_RANDOM_PIVOT 3"
  "CUDD_REORDER_SIFT 4"
  "CUDD_REORDER_SIFT_CONVERGE 5"
  "CUDD_REORDER_SYMM_SIFT 6"
  "CUDD_REORDER_SYMM_SIFT_CONV 7"
  "CUDD_REORDER_WINDOW2 8"
  "CUDD_REORDER_WINDOW3 9"
  "CUDD_REORDER_WINDOW4 10"
  "CUDD_REORDER_WINDOW2_CONV 11"
  "CUDD_REORDER_WINDOW3_CONV 12"
  "CUDD_REORDER_WINDOW4_CONV 13"
  "CUDD_REORDER_GROUP_SIFT 14"
  "CUDD_REORDER_GROUP_SIFT_CONV 15"
  "CUDD_REORDER_ANNEALING 16"
  "CUDD_REORDER_GENETIC 17"
  "CUDD_REORDER_LAZY_SIFT 20"
  "CUDD_REORDER_EXACT 21"
)

# Ensure the program is compiled
make clean
make

# Loop over all .blif files in the folder and subfolders
find "$BLIF_FOLDER" -type f -name "*.blif" | while read -r BLIF_FILE; do
    # Extract the circuit name from the BLIF file path
    CIRCUIT_NAME=$(basename "$BLIF_FILE" .blif)

    # Set the output directory based on the circuit name
    OUTPUT_DIR="${CIRCUIT_NAME}_BDD_all"

    # Create an output directory based on the circuit name
    mkdir -p "$OUTPUT_DIR"

    # Initialize variables to track the best algorithm and the smallest BDD size
    best_algorithm=""
    smallest_dag_size=""
    best_algorithm_output=""

    # Loop over each algorithm and test it
    for ALGO in "${ALGORITHMS[@]}"; do
        algo_name=$(echo $ALGO | awk '{print $1}')
        algo_value=$(echo $ALGO | awk '{print $2}')

        echo "Testing algorithm: $algo_name on $CIRCUIT_NAME..."

        # Define the output file for this algorithm
        OUTPUT_FILE="${OUTPUT_DIR}/${algo_name}_output.txt"

        # Run the program with a time limit of 20 minutes and capture output
        gtimeout 8m ./optimalAlgorithm "$BLIF_FILE" "$algo_value" > "$OUTPUT_FILE" 2>&1

        # Check if the program timed out
        if [ $? -eq 124 ]; then
            echo "Algorithm $algo_name timed out after 20 minutes." | tee -a "$OUTPUT_FILE"
            continue
        else
            echo "Completed: $algo_name." | tee -a "$OUTPUT_FILE"
        fi

        # Check if the output contains "Error". If so, skip this algorithm.
        if grep -q "Error" "$OUTPUT_FILE"; then
            echo "Skipping $algo_name due to an error in the output."
            continue
        fi

        # Extract the total DAG size from the output
        dag_size=$(grep "Total DAG size" "$OUTPUT_FILE" | awk '{print $NF}')

        # If we found a DAG size, compare it to find the smallest
        if [ ! -z "$dag_size" ]; then
            if [ -z "$smallest_dag_size" ] || [ "$dag_size" -lt "$smallest_dag_size" ]; then
                smallest_dag_size=$dag_size
                best_algorithm=$algo_name
                best_algorithm_output="$OUTPUT_FILE"
            fi
        fi

        echo "Results for $algo_name saved to $OUTPUT_FILE"
    done

    # Output the best algorithm to a text file
    if [ ! -z "$best_algorithm" ]; then
        echo "The algorithm that found the smallest BDD size for $CIRCUIT_NAME: $best_algorithm"
        echo "Smallest DAG size: $smallest_dag_size"
        echo "Best algorithm: $best_algorithm" > "${OUTPUT_DIR}/best_algorithm.txt"
        echo "Smallest DAG size: $smallest_dag_size" >> "${OUTPUT_DIR}/best_algorithm.txt"
        echo "Full output is available in $best_algorithm_output" >> "${OUTPUT_DIR}/best_algorithm.txt"
    else
        echo "No valid algorithms found without errors for $CIRCUIT_NAME."
        echo "No valid algorithms found without errors." > "${OUTPUT_DIR}/best_algorithm.txt"
    fi
done