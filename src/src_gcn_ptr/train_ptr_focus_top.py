# train_ptr_focus_top.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from blif_dataset import BlifDataset
from model_gcn_ptr import BDDVariableOrderingModel
import numpy as np
import random
import os
import traceback
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def parse_args():
    parser = argparse.ArgumentParser(description="Train BDD Variable Ordering Model")
    parser.add_argument('--train_samples', type=int, default=100000, help='Number of training samples to process per epoch')
    parser.add_argument('--val_samples', type=int, default=100000, help='Number of validation samples to process per epoch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--print_every', type=int, default=5, help='Frequency of printing loss and evaluations')
    args = parser.parse_args()
    return args

def set_random_seed(seed=42):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flatten(lst):
    """Recursively flattens a nested list and converts tensors to integers."""
    if isinstance(lst, torch.Tensor):
        return [lst.item()]  # Wrap the integer in a list
    if not isinstance(lst, list):
        return [lst]  # Wrap non-list items in a list
    if not lst:
        return lst
    flattened = []
    for item in lst:
        flattened.extend(flatten(item))  # Always extending with a list
    return flattened

class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    graphs, rank_wise_labels_list = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, rank_wise_labels_list

def main():
    print("Starting training script.")

    # Parse command-line arguments
    args = parse_args()

    # Set random seed for reproducibility
    set_random_seed(42)
    print("Random seed set.")

    # Load the training dataset
    print("Loading training dataset...")
    train_dataset = BlifDataset('../data/processed/final_train_dataset.pt')
    print(f"Training dataset loaded with {len(train_dataset)} samples.")

    # Check if training dataset is empty
    if len(train_dataset) == 0:
        print("[ERROR] Training dataset is empty.")
        return

    # Load the validation dataset
    print("Loading validation dataset...")
    val_dataset = BlifDataset('../data/processed/final_val_dataset.pt')  # Ensure correct path
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")

    # Check if validation dataset is empty
    if len(val_dataset) == 0:
        print("[WARNING] Validation dataset is empty. Proceeding without validation.")
        use_validation = False
    else:
        use_validation = True

    # Define the number of training samples to process
    num_train_samples = min(args.train_samples, len(train_dataset))
    print(f"Number of training samples to process per epoch: {num_train_samples}")

    # Get the first num_train_samples training samples
    print(f"Fetching the first {num_train_samples} samples from the training dataset...")
    train_samples = []
    for i in range(num_train_samples):
        try:
            graph, rank_wise_labels = train_dataset[i]
            train_samples.append((graph, rank_wise_labels))
            print(f"Fetched training sample {i+1}.")
        except IndexError:
            print(f"[WARNING] Training dataset has fewer than {num_train_samples} samples.")
            break

    # Get the first num_val_samples validation samples (if available)
    if use_validation:
        num_val_samples = min(args.val_samples, len(val_dataset))
        print(f"Number of validation samples to process per epoch: {num_val_samples}")
        print(f"Fetching the first {num_val_samples} samples from the validation dataset...")
        val_samples = []
        for i in range(num_val_samples):
            try:
                graph, rank_wise_labels = val_dataset[i]
                val_samples.append((graph, rank_wise_labels))
                print(f"Fetched validation sample {i+1}.")
            except IndexError:
                print(f"[WARNING] Validation dataset has fewer than {num_val_samples} samples.")
                break
    else:
        val_samples = []

    # Determine the maximum number of input nodes across all samples
    max_num_input_nodes = 0
    for graph, _ in train_samples + val_samples:
        node_types = graph.ndata['feat'][:, 1].long()
        num_input_nodes = (node_types == 0).sum().item()
        if num_input_nodes > max_num_input_nodes:
            max_num_input_nodes = num_input_nodes

    print(f"Maximum number of input nodes across all samples: {max_num_input_nodes}")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get input feature size
    if num_train_samples > 0:
        in_feats = train_samples[0][0].ndata['feat'].shape[1]
        print(f"Input feature size: {in_feats}")
    else:
        print("[ERROR] No training samples available.")
        return

    # Determine the number of classes
    num_classes = max_num_input_nodes + 1  # +1 for the <end> token
    print(f"Number of classes (num_classes): {num_classes}")

    # Initialize the model with the determined number of classes
    print(f"Initializing the model with {num_classes} classes...")
    model = BDDVariableOrderingModel(in_feats).to(device)
    print("Model initialized.")

    # Set up optimizer and loss function
    print("Setting up optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Optimizer and loss function set.")

    # Training parameters
    num_epochs = args.epochs
    teacher_forcing_ratio = 1.0  # Start with full teacher forcing
    print_every = args.print_every  # Print loss every 'print_every' epochs

    # Define directory to save best model
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pth')

    # Initialize best losses to infinity
    best_val_loss = float('inf')  # Only keep best validation loss

    # Create a batched DataLoader with a custom collate function
    train_dataset_loader = DataLoader(
        CustomDataset(train_samples),
        batch_size=4,  # Adjust batch_size as needed
        shuffle=True,
        num_workers=8,  
        collate_fn=collate_fn
    )

    # Similarly for validation
    val_dataset_loader = DataLoader(
        CustomDataset(val_samples),
        batch_size=4,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

    # Training loop
    print("Starting training loop...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0  # For tracking average loss
        num_processed_samples = 0
        
        # Iterate over the training batches
        for batch_idx, (batched_graph, rank_wise_labels_batch) in enumerate(train_dataset_loader):
            try:
                batched_graph = batched_graph.to(device)
                node_features = batched_graph.ndata['feat'].to(device)
                node_types = node_features[:, 1].long()

                # Prepare target sequences
                target_sequences = []
                for rank_wise_labels in rank_wise_labels_batch:
                    if isinstance(rank_wise_labels, torch.Tensor):
                        rank_wise_labels = rank_wise_labels.tolist()
                    target_sequence_node_indices = rank_wise_labels.copy()
                    if target_sequence_node_indices[-1] != -1:
                        target_sequence_node_indices.append(-1)  # Append end token

                    target_sequences.append(torch.tensor(target_sequence_node_indices, dtype=torch.long))

                # Pad target sequences
                target_sequence_batch = pad_sequence(target_sequences, batch_first=True, padding_value=-1).to(device)

                # Forward pass
                outputs, selected_indices = model(
                    batched_graph, node_features, node_types, target_sequence=target_sequence_batch, teacher_forcing_ratio=teacher_forcing_ratio
                )

                if outputs is None:
                    print(f"[WARNING] Model returned None outputs for batch {batch_idx + 1}. Skipping.")
                    continue

                # Reshape outputs and targets for loss computation
                batch_size, sequence_length, num_classes = outputs.size()
                outputs = outputs.view(batch_size, sequence_length, num_classes)
                target_sequence_batch = target_sequence_batch.view(batch_size, sequence_length)

                # Compute loss per token with reduction='none'
                loss_fn = nn.NLLLoss(reduction='none', ignore_index=-1)
                loss_per_token = loss_fn(
                    outputs.view(-1, num_classes), target_sequence_batch.view(-1)
                ).view(batch_size, sequence_length)

                # Compute valid mask
                valid_mask = (target_sequence_batch != -1).float()

                # Compute positional weights
                weights = torch.ones(sequence_length, device=device)
                top_k = max(1, int(sequence_length * 0.3))  # Top 20% positions
                weights[:top_k] = 2.0  # Assign higher weight to top positions
                weights = weights.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, sequence_length]

                # Apply weights and valid mask
                loss_per_token_weighted = loss_per_token * weights * valid_mask

                # Sum loss per sample and normalize
                loss_per_sample = loss_per_token_weighted.sum(dim=1)  # Shape: [batch_size]
                sequence_lengths = valid_mask.sum(dim=1)  # Shape: [batch_size]
                epsilon = 1e-8
                loss_per_sample_normalized = loss_per_sample / (sequence_lengths + epsilon)

                # Compute mean loss over the batch
                loss = loss_per_sample_normalized.mean()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()
                num_processed_samples += 1  # Each batch is counted as one sample

            except Exception as e:
                print(f"[ERROR] Exception during processing batch {batch_idx + 1}: {e}")
                traceback.print_exc()
                continue

        if num_processed_samples == 0:
            print(f"[WARNING] No training samples processed in epoch {epoch}.")
            continue

        # Average training loss for the epoch
        avg_epoch_loss = epoch_loss / num_processed_samples
        print(f"Epoch [{epoch}/{num_epochs}], Average Training Loss: {avg_epoch_loss:.4f}")

        # Logging and evaluation
        if epoch % print_every == 0:
            # Evaluate on validation data
            if use_validation and len(val_samples) > 0:
                model.eval()
                validation_loss = 0.0
                num_processed_samples = 0
                with torch.no_grad():
                    print(f"\nEpoch {epoch}: Evaluating model predictions on validation data...")
                    for batch_idx, (batched_graph, rank_wise_labels_batch) in enumerate(val_dataset_loader):
                        try:
                            batched_graph = batched_graph.to(device)
                            node_features = batched_graph.ndata['feat'].to(device)
                            node_types = node_features[:, 1].long()

                            # Prepare target sequences
                            target_sequences = []
                            for rank_wise_labels in rank_wise_labels_batch:
                                if isinstance(rank_wise_labels, torch.Tensor):
                                    rank_wise_labels = rank_wise_labels.tolist()
                                target_sequence_node_indices = rank_wise_labels.copy()
                                if target_sequence_node_indices[-1] != -1:
                                    target_sequence_node_indices.append(-1)  # Append end token

                                target_sequences.append(torch.tensor(target_sequence_node_indices, dtype=torch.long))

                            # Pad target sequences
                            target_sequence_batch = pad_sequence(target_sequences, batch_first=True, padding_value=-1).to(device)

                            # Forward pass
                            outputs, selected_indices = model(
                                batched_graph, node_features, node_types, target_sequence=target_sequence_batch, teacher_forcing_ratio=1.0
                            )

                            if outputs is None:
                                print(f"[WARNING] Model returned None outputs for validation batch {batch_idx + 1}. Skipping.")
                                continue

                            # Reshape outputs and targets for loss computation
                            batch_size, sequence_length, num_classes = outputs.size()
                            outputs = outputs.view(batch_size, sequence_length, num_classes)
                            target_sequence_batch = target_sequence_batch.view(batch_size, sequence_length)

                            # Compute loss per token with reduction='none'
                            loss_fn = nn.NLLLoss(reduction='none', ignore_index=-1)
                            loss_per_token = loss_fn(
                                outputs.view(-1, num_classes), target_sequence_batch.view(-1)
                            ).view(batch_size, sequence_length)

                            # Compute valid mask
                            valid_mask = (target_sequence_batch != -1).float()

                            # Compute positional weights
                            weights = torch.ones(sequence_length, device=device)
                            top_k = max(1, int(sequence_length * 0.2))  # Top 20% positions
                            weights[:top_k] = 1.8  # Assign higher weight to top positions
                            weights = weights.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, sequence_length]

                            # Apply weights and valid mask
                            loss_per_token_weighted = loss_per_token * weights * valid_mask

                            # Sum loss per sample and normalize
                            loss_per_sample = loss_per_token_weighted.sum(dim=1)  # Shape: [batch_size]
                            sequence_lengths = valid_mask.sum(dim=1)  # Shape: [batch_size]
                            epsilon = 1e-8
                            loss_per_sample_normalized = loss_per_sample / (sequence_lengths + epsilon)

                            # Compute mean loss over the batch
                            loss = loss_per_sample_normalized.mean()

                            # Accumulate validation loss
                            validation_loss += loss.item()
                            num_processed_samples += 1  # Each batch is counted as one sample

                        except Exception as e:
                            print(f"[ERROR] Exception during evaluation of validation batch {batch_idx + 1}: {e}")
                            traceback.print_exc()
                            continue

                    if num_processed_samples > 0:
                        avg_validation_loss = validation_loss / num_processed_samples
                        print(f"Epoch [{epoch}/{num_epochs}], Average Validation Loss: {avg_validation_loss:.4f}")

                        # Check if current epoch has the best (smallest) validation loss
                        if avg_validation_loss < best_val_loss:
                            best_val_loss = avg_validation_loss
                            # Save the model state_dict
                            torch.save(model.state_dict(), best_model_path)
                            print(f"*** New best model saved at epoch {epoch} with validation loss {avg_validation_loss:.4f} ***")
                    else:
                        print(f"No validation samples were processed in epoch {epoch}.")

                # Switch back to training mode
                model.train()

if __name__ == '__main__':
    main()