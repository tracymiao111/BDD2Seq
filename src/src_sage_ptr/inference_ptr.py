# inference_ptr.py

import torch
import torch.nn.functional as F
import dgl
import json
from blif_dataset_infer import BlifDataset
from model_sage_ptr import BDDVariableOrderingModel
import argparse
import os
import traceback

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
    parser = argparse.ArgumentParser(description="Inference Script for BDD Variable Ordering Model")
    parser.add_argument('--model_path', type=str, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, help='Path to the test dataset')
    parser.add_argument('--output_path', type=str, help='Path to save the inference results')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    print("Loading test dataset...")
    test_dataset = BlifDataset(args.data_path)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    # Check if test dataset is empty
    if len(test_dataset) == 0:
        print("[ERROR] Test dataset is empty.")
        return

    # Get input feature size
    in_feats = test_dataset[0][0].ndata['feat'].shape[1]
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
                graph, rank_wise_labels, circuit_name = test_dataset[sample_idx]
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
                node_idx_to_decoder_idx[-1] = end_token_decoder_idx  # Map end token
                decoder_idx_to_node_idx[end_token_decoder_idx] = -1

                # Prepare encoder outputs
                encoder_outputs = model.encoder(graph, node_features)  # Shape: [num_nodes, hidden_size]
                encoder_outputs = encoder_outputs[input_node_indices]  # Only input nodes
                encoder_outputs = encoder_outputs.unsqueeze(0)  # Add batch dimension

                # Prepare node_types for decoder (if needed)
                node_types = node_types[input_node_indices]

                # Run the decoder without teacher forcing
                outputs, selected_indices = model.decoder(
                    encoder_outputs, target_sequence=None, teacher_forcing_ratio=0.0
                )

                # Extract the predicted sequence
                selected_indices = selected_indices.squeeze(0).cpu().tolist()  # List of decoder indices
                predicted_node_indices = [decoder_idx_to_node_idx[idx] for idx in selected_indices]

                # Remove end token (-1)
                if -1 in predicted_node_indices:
                    end_idx = predicted_node_indices.index(-1)
                    predicted_node_indices = predicted_node_indices[:end_idx]

                # Prepare ground truth
                if isinstance(rank_wise_labels, torch.Tensor):
                    rank_wise_labels = rank_wise_labels.tolist()
                else:
                    rank_wise_labels = flatten(rank_wise_labels)

                # Filter out node indices with node_types != 0
                valid_ground_truth = [idx for idx in rank_wise_labels if node_types[node_idx_to_decoder_idx[idx]] == 0]

                # Store the result
                result = {
                    "circuit_name": circuit_name,
                    "predicted_ordering": predicted_node_indices,
                    "ground_truth_ordering": valid_ground_truth
                }
                results.append(result)
                print(f"Processed sample {sample_idx + 1}: {circuit_name}")

            except Exception as e:
                print(f"[ERROR] Exception during inference of sample {sample_idx + 1}: {e}")
                traceback.print_exc()
                continue

    # Save results to JSON file
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Inference results saved to {args.output_path}")

if __name__ == '__main__':
    main()