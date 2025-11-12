import os
import re
import random
from pathlib import Path

def extract_input_signals(verilog_code):
    # Extract the input signals from the Verilog code
    inputs_match = re.search(r'input\s+(.*?);', verilog_code, re.DOTALL)
    if inputs_match:
        # Find all inputs, including escaped identifiers (e.g., \1GAT(0)) and normal inputs (e.g., N1)
        input_signals = re.findall(r'\\?\w+\(?\d*\)?', inputs_match.group(1))
        return input_signals
    return []

def create_negated_wires(input_signals, negate_count):
    # Ensure that the number of negations does not exceed the number of available input signals
    negate_count = min(negate_count, len(input_signals))
    
    # Randomly select input signals to negate
    inputs_to_negate = random.sample(input_signals, negate_count)
    
    # Create new wire definitions and negation assignments with "~"
    new_wires = [f'wire {sig}_neg ;' for sig in inputs_to_negate]
    negations = [f'assign {sig}_neg = ~{sig} ;' for sig in inputs_to_negate]
    
    return new_wires, negations, inputs_to_negate

def replace_negations_in_assign(verilog_code, inputs_to_negate):
    # Replace instances of selected input signals with their negated versions in assign statements
    def replace_neg(match):
        sig = match.group(0)
        if sig in inputs_to_negate:
            return f'{sig}_neg'
        return sig
    
    assign_match = re.findall(r'assign\s+.*?;', verilog_code, re.DOTALL)
    for assign in assign_match:
        updated_assign = re.sub(r'\\?\w+\(?\d*\)?', replace_neg, assign)
        verilog_code = verilog_code.replace(assign, updated_assign)
    
    return verilog_code

def insert_negated_wires(verilog_code, new_wires, negations):
    # Insert new wires and negation assignments after the last 'wire' declaration
    insertion_point = verilog_code.rfind('wire')
    if insertion_point != -1:
        insertion_point = verilog_code.find(';', insertion_point) + 1
        new_verilog = verilog_code[:insertion_point] + '\n' + '\n'.join(new_wires) + '\n' + '\n'.join(negations) + '\n' + verilog_code[insertion_point:]
    else:
        # If no 'wire' declaration found, insert before the 'assign' section
        insertion_point = verilog_code.find('assign')
        new_verilog = verilog_code[:insertion_point] + '\n' + '\n'.join(new_wires) + '\n' + '\n'.join(negations) + '\n' + verilog_code[insertion_point:]
    
    return new_verilog

def process_verilog_file(filepath, output_dir, num_mutations):
    # Read the Verilog file
    with open(filepath, 'r') as file:
        verilog_code = file.read()

    # Extract input signals
    input_signals = extract_input_signals(verilog_code)
    if not input_signals:
        print(f"No input signals found in {filepath}")
        return

    # Ensure the number of mutations doesn't exceed the number of input signals
    max_mutations = min(num_mutations, len(input_signals))

    # Generate multiple distinct mutations for 1 to 10 negations
    for negate_count in range(1, 11):
        for mutation_idx in range(1, max_mutations + 1):
            new_wires, negations, inputs_to_negate = create_negated_wires(input_signals, negate_count)
            modified_code = replace_negations_in_assign(verilog_code, inputs_to_negate)
            modified_code = insert_negated_wires(modified_code, new_wires, negations)

            # Save the modified Verilog file with a distinct mutation index
            file_stem = Path(filepath).stem.replace('_orig', '')
            output_filename = f"{file_stem}_n{negate_count}_m{mutation_idx}.v"
            output_filepath = os.path.join(output_dir, output_filename)
            
            with open(output_filepath, 'w') as output_file:
                output_file.write(modified_code)
            print(f"Generated: {output_filename}")

def main(folder_path, num_mutations=10):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(folder_path, 'negated_circuits')
    os.makedirs(output_dir, exist_ok=True)

    # Process each .v file named xxxx_orig.v in the folder
    for verilog_file in Path(folder_path).glob('*_orig.v'):
        print(f"Processing: {verilog_file.name}")
        process_verilog_file(verilog_file, output_dir, num_mutations)

if __name__ == "__main__":
    folder_path = ''  # Replace with your folder path
    num_mutations = 5  # Set how many distinct mutations you want for each negation count
    main(folder_path, num_mutations)