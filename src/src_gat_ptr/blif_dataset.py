# blif_dataset.py
import torch
from torch.utils.data import Dataset
import dgl

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

class BlifDataset(Dataset):
    def __init__(self, pt_file):
        """
        Initializes the BlifDataset.

        Args:
            pt_file (str): Path to the .pt file containing the dataset.
        """
        data = torch.load(pt_file)
        
        if isinstance(data, dict):
            if 'graphs' in data and 'labels' in data:
                self.graphs = data['graphs']  # List of DGLGraph objects
                self.rank_wise_labels = data['labels']  # List of lists or tensors
            else:
                raise KeyError("The .pt file must contain 'graphs' and 'labels' keys.")
        else:
            raise ValueError("Unsupported data format in .pt file. Expected a dictionary with 'graphs' and 'labels' keys.")
        
        print(f"Number of graphs: {len(self.graphs)}")
        print(f"Number of label sets: {len(self.rank_wise_labels)}")

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        rank_wise_labels = self.rank_wise_labels[idx]
        
        # Clone 'feat' to preserve it before modifications
        if 'feat' not in graph.ndata:
            print(f"[ERROR] 'feat' not found in graph at index {idx} BEFORE modification.")
            print(f"Available node data keys: {list(graph.ndata.keys())}")
            raise KeyError(f"'feat' not found in node data for graph at index {idx} BEFORE modification.")
        
        feat = graph.ndata['feat'].clone()
        
        # Convert to bidirected graph and add self-loops
        graph = dgl.to_bidirected(graph)
        graph = dgl.add_self_loop(graph)
        
        # Reassign 'feat' to ensure it's preserved after modifications
        graph.ndata['feat'] = feat
        
        # Validate 'feat' after modification
        if 'feat' not in graph.ndata:
            print(f"[ERROR] 'feat' not found in graph at index {idx} AFTER modification.")
            print(f"Available node data keys: {list(graph.ndata.keys())}")
            raise KeyError(f"'feat' not found in node data for graph at index {idx} AFTER modification.")
        
        # Ensure rank_wise_labels is a flat list of integers and corresponds to node_types == 0
        if isinstance(rank_wise_labels, torch.Tensor):
            rank_wise_labels = rank_wise_labels.tolist()
        elif isinstance(rank_wise_labels, list):
            # Flatten if necessary
            rank_wise_labels = flatten(rank_wise_labels)
        else:
            raise TypeError(f"Unsupported label type: {type(rank_wise_labels)}")
        
        # Filter out node indices with node_types != 0
        node_types = graph.ndata['feat'][:, 1].long()
        valid_node_indices = [node_idx for node_idx in rank_wise_labels if node_types[node_idx] == 0]
        
        # **Check for duplicates**
        if len(valid_node_indices) != len(set(valid_node_indices)):
            duplicates = set([x for x in valid_node_indices if valid_node_indices.count(x) > 1])
            print(f"[ERROR] Duplicate node indices found in labels at index {idx}: {duplicates}")
            raise ValueError(f"Duplicate node indices in labels for graph {idx}: {duplicates}")
        
        return graph, valid_node_indices