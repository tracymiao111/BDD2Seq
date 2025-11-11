import os
import glob
from itertools import product
import torch
import networkx as nx
import random
import blifparser.blifparser as blifparser
import dgl
import time  

def truth_table_vector(boolfunc):
    """
    Generates the truth table vector for a given boolean function.
    """
    inputs = boolfunc.inputs
    truthtable = boolfunc.truthtable  # Each row is a list, including inputs and output

    num_inputs = len(inputs)
    # Generate all possible input combinations in lex order
    input_combinations = list(product(['0', '1'], repeat=num_inputs))

    truth_table_vector = []

    for comb in input_combinations:
        output = '0'  # Default output is '0' unless specified
        # Check if comb matches any row in the truth table
        for row in truthtable:
            row_inputs = row[:-1]  # Inputs in the row
            row_output = row[-1]   # Output in the row

            match = True
            for ci, ri in zip(comb, row_inputs):
                if ri == '-':
                    continue  # Don't-care matches any input
                elif ci != ri:
                    match = False
                    break  # Inputs don't match; move to next row
            if match:
                output = row_output
                break  # Stop at the first matching row

        truth_table_vector.append(int(output))

    return truth_table_vector

def process_blif_file(filepath, padding_length=8):
    """
    Processes a BLIF file and returns a DGLGraph, including additional node features.
    """
    # Parse the BLIF file
    parser = blifparser.BlifParser(filepath)
    blif = parser.blif

    # Get the circuit name from the file path
    circuit_name = os.path.splitext(os.path.basename(filepath))[0]

    # Access the lists of input and output names
    inputs_list = blif.inputs.inputs  # List of input names
    outputs_list = blif.outputs.outputs  # List of output names

    nodes = []
    node_name_to_indices = {}  # Map node names to a dictionary of types to indices

    # Assign indices starting from 0
    node_index = 0

    # Process inputs
    for input_name in inputs_list:
        node = {
            'index': node_index,
            'name': input_name,
            'type': 0,  # Input encoded as integer 0
            'feature_vector': [node_index, 0] + [0]*padding_length  # [Index, Type, zeros]
        }
        nodes.append(node)
        # Update node_name_to_indices
        if input_name not in node_name_to_indices:
            node_name_to_indices[input_name] = {}
        node_name_to_indices[input_name][0] = node_index  # Type 0 is input
        node_index += 1

    # Process gates
    for boolfunc in blif.booleanfunctions:
        tt_vector = truth_table_vector(boolfunc)
        # Pad or trim the tt_vector to length padding_length
        if len(tt_vector) < padding_length:
            tt_vector_padded = tt_vector + [0]*(padding_length - len(tt_vector))
        else:
            tt_vector_padded = tt_vector[:padding_length]
        node = {
            'index': node_index,
            'name': boolfunc.output,
            'type': 1,  # Gate encoded as integer 1
            'feature_vector': [node_index, 1] + tt_vector_padded
        }
        nodes.append(node)
        if node['name'] not in node_name_to_indices:
            node_name_to_indices[node['name']] = {}
        node_name_to_indices[node['name']][1] = node_index  # Type 1 is gate
        node_index += 1

    # Process outputs
    for output_name in outputs_list:
        node = {
            'index': node_index,
            'name': output_name,
            'type': 2,  # Output encoded as integer 2
            'feature_vector': [node_index, 2] + [0]*padding_length  # [Index, Type, zeros]
        }
        nodes.append(node)
        if output_name not in node_name_to_indices:
            node_name_to_indices[output_name] = {}
        node_name_to_indices[output_name][2] = node_index  # Type 2 is output
        node_index += 1

    # Construct edges
    edges = []

    # For each gate, create edges from its inputs to the gate node
    for boolfunc in blif.booleanfunctions:
        gate_node_index = node_name_to_indices[boolfunc.output][1]  # Type 1 is gate
        for input_name in boolfunc.inputs:
            # Get the index of the input node
            input_node_index = None
            if input_name in node_name_to_indices:
                if 0 in node_name_to_indices[input_name]:
                    input_node_index = node_name_to_indices[input_name][0]  # Input node
                elif 1 in node_name_to_indices[input_name]:
                    input_node_index = node_name_to_indices[input_name][1]  # Gate output node
                else:
                    # Maybe it's an output node (unlikely as input to a gate)
                    input_node_index = node_name_to_indices[input_name].get(2)
            if input_node_index is None:
                raise ValueError(f"Input node '{input_name}' not found in node_name_to_indices")
            # Add edge from input node to gate node
            edges.append((input_node_index, gate_node_index))

    # For each output, create an edge from the node producing it to the output node
    for output_name in outputs_list:
        output_node_index = node_name_to_indices[output_name][2]  # Type 2 is output
        # Find the node that produces this output
        producer_node_index = None
        if output_name in node_name_to_indices:
            if 1 in node_name_to_indices[output_name]:
                producer_node_index = node_name_to_indices[output_name][1]  # Gate node
            elif 0 in node_name_to_indices[output_name]:
                producer_node_index = node_name_to_indices[output_name][0]  # Input node
            else:
                raise ValueError(f"No producer node found for output '{output_name}'")
        else:
            raise ValueError(f"Output node '{output_name}' not found in node_name_to_indices")
        # Add edge from producer node to output node
        edges.append((producer_node_index, output_node_index))

    # Create a NetworkX directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes:
        G.add_node(
            node['index'],
            name=node['name'],
            type=node['type'],
            feature_vector=node['feature_vector']
        )

    # Add edges to the graph
    G.add_edges_from(edges)

    # Fan-in and Fan-out
    fan_in = dict(G.in_degree())
    fan_out = dict(G.out_degree())

    # Topological ordering
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print(f"[ERROR] Graph {circuit_name} is not a DAG.")
        return None, circuit_name
    for idx, node_id in enumerate(topo_order):
        G.nodes[node_id]['topo_order'] = idx  # Add topo_order as node attribute

    # Calculating depth
    depths = {node_id: 0 for node_id in G.nodes()}
    for node_id in topo_order:
        current_depth = depths[node_id]
        G.nodes[node_id]['depth'] = current_depth  # Add depth as node attribute
        for successor in G.successors(node_id):
            if depths[successor] < current_depth + 1:
                depths[successor] = current_depth + 1

    # Add fan-in and fan-out as node attributes
    for node_id in G.nodes():
        G.nodes[node_id]['fan_in'] = fan_in[node_id]
        G.nodes[node_id]['fan_out'] = fan_out[node_id]

    # Initialize lists to store new features
    feature_vectors = []
    node_indices = []
    node_names = []

    # Update feature vectors with new information
    for node in nodes:
        node_id = node['index']
        # Retrieve additional information from the graph
        topo_order_val = G.nodes[node_id]['topo_order']
        depth = G.nodes[node_id]['depth']
        fan_in_value = G.nodes[node_id]['fan_in']
        fan_out_value = G.nodes[node_id]['fan_out']
        # Original feature vector (already includes node_index and node_type)
        original_feature_vector = node['feature_vector']
        # Remove node_index and node_type from original_feature_vector as they are already included
        feature_vector_padding = original_feature_vector[2:]  # Exclude index and type

        # Create new feature vector
        new_feature_vector = [
            node['index'],       # Node index
            node['type'],        # Node type
            topo_order_val,      # Topological order
            depth,               # Depth
            fan_in_value,        # Fan-in
            fan_out_value        # Fan-out
        ] + feature_vector_padding  # Append the existing feature vector (padded truth table vector)

        # Update the node's feature vector
        node['feature_vector'] = new_feature_vector

        # Collect features for tensors
        feature_vectors.append(new_feature_vector)
        node_indices.append(node['index'])
        node_names.append(node['name'])

    # Convert lists to tensors
    node_features_tensor = torch.tensor(feature_vectors, dtype=torch.float)  # Shape: [num_nodes, feature_dim]
    node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)       # Shape: [num_nodes]

    # Convert edge list to tensors
    src_nodes = [edge[0] for edge in edges]
    dst_nodes = [edge[1] for edge in edges]

    # Create a DGLGraph
    g = dgl.graph((src_nodes, dst_nodes))

    # Assign node features to the graph
    g.ndata['feat'] = node_features_tensor
    if 'feat' not in g.ndata:
        print(f"[ERROR] Graph from {circuit_name} does not have 'feat' assigned.")
        return None, circuit_name

    # Return the graph and circuit name
    return g, circuit_name

def main_processing():
    # Folder containing BLIF files
    blif_folder = "enter the path here."  # Update this path

    # Get list of BLIF files
    blif_files = glob.glob(os.path.join(blif_folder, "*.blif"))

    # Sort the files for consistency (optional)
    blif_files.sort()

    # Initialize lists to store graph data
    graphs = []  # List to store DGLGraphs
    circuit_names = []  # To keep track of circuit names
    total_time = 0  # Variable to track the total time

    for filepath in blif_files:
        try:
            start_time = time.time()  # Start time for the current sample

            print(f"Processing {os.path.basename(filepath)}")
            g, circuit_name = process_blif_file(filepath)

            if g is None:
                print(f"[SKIPPED] {os.path.basename(filepath)} due to errors in processing.")
                continue  # Skip this graph

            # Store the graph and circuit name
            graphs.append(g)
            circuit_names.append(circuit_name)

            # Record end time for current sample
            end_time = time.time()
            time_taken = end_time - start_time
            total_time += time_taken  # Add the time for this sample to the total
            print(f"Processed graph for {circuit_name}")
            print(f"Time taken for {circuit_name}: {time_taken:.2f} seconds")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    # Check if any graphs were processed
    if not graphs:
        print("No graphs were processed successfully.")
        return

    # Print the total time for all samples
    print(f"Total time for processing all samples: {total_time:.2f} seconds")

    # Create a dataset dictionary
    dataset = {
        'graphs': graphs,
        'circuit_names': circuit_names  # Include circuit names
    }

    # Save the dataset as a dictionary
    save_dir = "enter the path to save"  # Replace with your desired directory path

    # Make sure the directory exists, if not, create it
    os.makedirs(save_dir, exist_ok=True)

    # Save the dataset
    dataset_path = os.path.join(save_dir, 'test_time.pt')
    torch.save(dataset, dataset_path)

    print(f"Dataset saved at {dataset_path}.")

if __name__ == '__main__':
    main_processing()