#!/usr/bin/env python3
"""
inference_ptr_div_beam.py

An example inference script that:
 - Loads a trained model for BDD variable ordering,
 - Performs "diverse beam search" decoding,
 - Returns up to --max_sequences sequences per sample.

Requires:
  - blif_dataset_no_label.py (which defines BlifDataset),
  - model_ptr_div_beam.py (which defines BDDVariableOrderingModel and diverse_beam_search).
"""

import argparse
import os
import json
import traceback

import torch
import torch.nn.functional as F
import dgl

from blif_dataset_no_label import BlifDataset
from model_ptr_div_beam import BDDVariableOrderingModel

def flatten(lst):
    """Recursively flattens a nested list and converts tensors to integers."""
    if isinstance(lst, torch.Tensor):
        return [lst.item()]
    if not isinstance(lst, list):
        return [lst]
    if not lst:
        return lst
    flattened = []
    for item in lst:
        flattened.extend(flatten(item))
    return flattened

def main():
    parser = argparse.ArgumentParser(description="Inference Script for BDD Variable Ordering Model with Diversity Beam Search")
    parser.add_argument('--model_path', type=str, default='models/best_model_archive.pth', 
                        help='Path to the trained model.')
    parser.add_argument('--data_path', type=str, 
                    help='Path to the test dataset.')
    parser.add_argument('--output_path', type=str, 
                        help='Path to save the inference results as a JSON.')
    parser.add_argument('--beam_width', type=int, default=50, 
                        help='Beam width for beam search.')
    parser.add_argument('--num_groups', type=int, default=25, 
                        help='Number of groups for diverse beam search.')
    parser.add_argument('--diversity_strength', type=float, default=0.25, 
                        help='Diversity strength for diverse beam search.')
    parser.add_argument('--max_sequences', type=int, default=50,
                        help='Maximum number of final sequences to keep per sample.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    print("Loading test dataset...")
    test_dataset = BlifDataset(args.data_path)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    # Check if test dataset is empty
    if len(test_dataset) == 0:
        print("[ERROR] Test dataset is empty.")
        return

    # Figure out the feature dimension from the first sample
    data_sample = test_dataset[0]
    if len(data_sample) == 3:
        graph_sample, _, _ = data_sample
    elif len(data_sample) == 2:
        graph_sample, _ = data_sample
    else:
        raise ValueError("Unexpected data format in test_dataset.")

    in_feats = graph_sample.ndata['feat'].shape[1]
    print(f"Input feature size: {in_feats}")

    # Initialize the model
    print("Initializing the model...")
    model = BDDVariableOrderingModel(in_feats).to(device)
    print("Model initialized.")

    # Load the trained model
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"[ERROR] Model file not found at {args.model_path}")
        return

    model.eval()

    # Prepare to store results
    results = []

    with torch.no_grad():
        for sample_idx in range(len(test_dataset)):
            try:
                data = test_dataset[sample_idx]

                if len(data) == 3:
                    # We have (graph, rank_wise_labels, circuit_name)
                    graph, rank_wise_labels, circuit_name = data
                elif len(data) == 2:
                    # We have (graph, circuit_name)
                    graph, circuit_name = data
                    rank_wise_labels = None
                else:
                    raise ValueError("Unexpected data format returned from dataset.")

                graph = graph.to(device)
                node_features = graph.ndata['feat'].to(device)

                # Identify input nodes (node_type == 0)
                node_types = node_features[:, 1].long()
                input_node_indices = torch.where(node_types == 0)[0]

                if input_node_indices.numel() == 0:
                    print(f"[WARNING] Sample {sample_idx + 1} ({circuit_name}) has no input nodes. Skipping.")
                    continue

                # Create mappings between node indices and decoder indices for this sample
                node_idx_to_decoder_idx = {node_idx.item(): idx for idx, node_idx in enumerate(input_node_indices)}
                decoder_idx_to_node_idx = {idx: node_idx.item() for idx, node_idx in enumerate(input_node_indices)}
                end_token_decoder_idx = len(node_idx_to_decoder_idx)
                # Map the <end> token
                node_idx_to_decoder_idx[-1] = end_token_decoder_idx
                decoder_idx_to_node_idx[end_token_decoder_idx] = -1

                # Encoder forward pass: [num_nodes, hidden_size]
                encoder_outputs = model.encoder(graph, node_features)
                # Only keep the inputs
                encoder_outputs = encoder_outputs[input_node_indices]  # shape [num_inputs, hidden_size]
                # Add batch dimension => shape [1, num_inputs, hidden_size]
                encoder_outputs = encoder_outputs.unsqueeze(0)

                # Run the decoder with diverse beam search
                beam_width = args.beam_width
                num_groups = args.num_groups
                diversity_strength = args.diversity_strength

                selected_sequences = model.decoder.diverse_beam_search(
                    encoder_outputs, 
                    beam_width=beam_width, 
                    num_groups=num_groups, 
                    diversity_strength=diversity_strength
                )

                # If your diverse_beam_search always returns only top `beam_width`,
                # you might see fewer. 
                # If it returns all sequences, you might see many. 
                # We'll do a final slice to keep up to args.max_sequences:
                if len(selected_sequences) > args.max_sequences:
                    selected_sequences = selected_sequences[:args.max_sequences]

                # Extract the predicted sequences
                predicted_orderings = []
                for selected_indices in selected_sequences:
                    selected_indices = selected_indices.cpu().tolist()  # List of decoder indices
                    predicted_node_indices = [decoder_idx_to_node_idx[idx] for idx in selected_indices]

                    # Remove end token (-1) if present
                    if -1 in predicted_node_indices:
                        end_idx = predicted_node_indices.index(-1)
                        predicted_node_indices = predicted_node_indices[:end_idx]

                    predicted_orderings.append(predicted_node_indices)

                # If we have ground-truth labels, flatten them
                valid_ground_truth = None
                if rank_wise_labels is not None:
                    if isinstance(rank_wise_labels, torch.Tensor):
                        rank_wise_labels = rank_wise_labels.tolist()
                    else:
                        rank_wise_labels = flatten(rank_wise_labels)

                    # Filter out node indices with node_types != 0
                    valid_ground_truth = [idx for idx in rank_wise_labels if idx in node_idx_to_decoder_idx]

                # Prepare the result dictionary
                result = {
                    "circuit_name": circuit_name,
                    "predicted_orderings": predicted_orderings
                }
                if valid_ground_truth is not None:
                    result["ground_truth_ordering"] = valid_ground_truth

                results.append(result)
                print(f"Processed sample {sample_idx + 1}: {circuit_name}")

            except Exception as e:
                print(f"[ERROR] Exception during inference of sample {sample_idx + 1}: {e}")
                traceback.print_exc()
                continue

    # Save results to JSON file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Inference results saved to {args.output_path}")

if __name__ == '__main__':
    main()