import torch
import torch.nn.functional as F
import dgl
import json
from blif_dataset_no_label import BlifDataset
from model_ptr import BDDVariableOrderingModel
import argparse
import os
import traceback
import time 

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
    parser.add_argument('--model_path', type=str, default='models/best_model_archive.pth', help='Path to the trained model')
    parser.add_argument('--data_path', type=str, help='Path to the test dataset')
    parser.add_argument('--output_path', type=str, help='Path to save the inference results')
    parser.add_argument('--gpu', type=int, default=1, help='GPU id to use, -1 for CPU')
    args = parser.parse_args()

    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

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
    total_time = 0  # Variable to track total time for all samples

    with torch.no_grad():
        for sample_idx in range(len(test_dataset)):
            try:
                start_time = time.perf_counter()  # Start time for the current sample

                # Check if labels exist for the sample
                data = test_dataset[sample_idx]
                if len(data) == 3:  # Labels available
                    graph, rank_wise_labels, circuit_name = data
                    has_labels = True
                elif len(data) == 2:  # No labels available
                    graph, circuit_name = data
                    rank_wise_labels = None
                    has_labels = False
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
                node_idx_to_decoder_idx[-1] = end_token_decoder_idx  # Map end token
                decoder_idx_to_node_idx[end_token_decoder_idx] = -1

                # Prepare encoder outputs
                encoder_outputs = model.encoder(graph, node_features)  # Shape: [num_nodes, hidden_size]
                encoder_outputs = encoder_outputs[input_node_indices]  # Only input nodes
                encoder_outputs = encoder_outputs.unsqueeze(0)  # Add batch dimension

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

                # If labels exist, process them
                valid_ground_truth = None
                if has_labels:
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
                    "ground_truth_ordering": valid_ground_truth if valid_ground_truth is not None else []
                }
                results.append(result)

                # End time for the current sample
                end_time = time.perf_counter()

                # Time consumed for the current sample
                sample_time = end_time - start_time
                total_time += sample_time  # Add to the total time

                print(f"Processed sample {sample_idx + 1}: {circuit_name}")
                print(f"Time consumed for sample {circuit_name}: {sample_time:.4f} seconds")

            except Exception as e:
                print(f"[ERROR] Exception during inference of sample {sample_idx + 1}: {e}")
                traceback.print_exc()
                continue

    # Save results to JSON file
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Inference results saved to {args.output_path}")
    print(f"Total time consumed for all samples: {total_time:.4f} seconds")

if __name__ == '__main__':
    main()