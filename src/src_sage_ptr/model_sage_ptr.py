# model_gcn_ptr.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_feats, hidden_size, num_layers, dropout, aggregator_type='mean'):
        super(GraphSAGEEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Input layer
        self.layers.append(SAGEConv(
            in_feats, hidden_size, aggregator_type=aggregator_type, feat_drop=dropout, activation=F.relu
        ))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(
                hidden_size, hidden_size, aggregator_type=aggregator_type, feat_drop=dropout, activation=F.relu
            ))

        # Output layer
        self.layers.append(SAGEConv(
            hidden_size, hidden_size, aggregator_type=aggregator_type, feat_drop=dropout, activation=None
        ))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for l in range(self.num_layers):
            h = self.layers[l](g, h)
            if l != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h  # Shape: [num_nodes, hidden_size]

class PointerDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(PointerDecoder, self).__init__()
        self.hidden_size = hidden_size

        # Decoder LSTM cell
        self.decoder_cell = nn.LSTMCell(hidden_size, hidden_size)

        # Attention layers
        self.attn_W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)

        # Embeddings for <start> and <end> tokens
        self.start_token = nn.Parameter(torch.Tensor(hidden_size))
        self.end_token = nn.Parameter(torch.Tensor(hidden_size))

        self.init_parameters()

    def init_parameters(self):
        nn.init.uniform_(self.start_token, -0.1, 0.1)
        nn.init.uniform_(self.end_token, -0.1, 0.1)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, encoder_outputs, target_sequence=None, teacher_forcing_ratio=0.5):
        device = encoder_outputs.device
        batch_size, num_input_nodes, hidden_size = encoder_outputs.size()

        # Include end token embedding
        end_token_embedding = self.end_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, hidden_size)
        extended_encoder_outputs = torch.cat([encoder_outputs, end_token_embedding], dim=1)  # Shape: [batch_size, num_input_nodes + 1, hidden_size]

        # Initialize decoder inputs
        decoder_input = self.start_token.unsqueeze(0).expand(batch_size, hidden_size)
        hx = torch.zeros(batch_size, self.hidden_size, device=device)
        cx = torch.zeros(batch_size, self.hidden_size, device=device)

        # Maximum decoding steps
        if target_sequence is not None:
            max_decoding_steps = target_sequence.size(1)
        else:
            max_decoding_steps = num_input_nodes + 1  # +1 for <end> token

        outputs = []
        selected_indices = []

        # Prepare masks for attention
        mask = torch.zeros(batch_size, num_input_nodes + 1, device=device)

        for t in range(max_decoding_steps):
            # Decoder LSTM cell
            hx, cx = self.decoder_cell(decoder_input, (hx, cx))

            # Attention mechanism
            attn_scores = self.attn_v(torch.tanh(
                self.attn_W_a(extended_encoder_outputs) + self.attn_U_a(hx).unsqueeze(1)
            )).squeeze(-1)  # Shape: [batch_size, num_input_nodes + 1]

            # Apply mask
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))

            # Subtract max for numerical stability
            attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]

            # Compute log probabilities
            log_probs = F.log_softmax(attn_scores, dim=1)  # Shape: [batch_size, num_input_nodes + 1]

            # Determine next input
            if target_sequence is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_idx = target_sequence[:, t]  # Shape: [batch_size]
            else:
                next_idx = log_probs.argmax(dim=1)  # Shape: [batch_size]

            # Update mask after prediction, only for indices less than num_input_nodes
            mask_indices = next_idx < num_input_nodes
            batch_indices = torch.arange(batch_size, device=device)
            mask[batch_indices[mask_indices], next_idx[mask_indices]] = 1

            # Prepare next decoder input
            decoder_input = torch.zeros(batch_size, hidden_size, device=device)
            for b in range(batch_size):
                idx = next_idx[b]
                if idx < num_input_nodes:
                    decoder_input[b] = encoder_outputs[b, idx]
                else:
                    decoder_input[b] = self.end_token

            outputs.append(log_probs)
            selected_indices.append(next_idx)

        outputs = torch.stack(outputs, dim=1)  # Shape: [batch_size, sequence_length, num_input_nodes + 1]
        selected_indices = torch.stack(selected_indices, dim=1)  # Shape: [batch_size, sequence_length]

        return outputs, selected_indices

class BDDVariableOrderingModel(nn.Module):
    def __init__(self, in_feats, hidden_size=512, num_layers=6, dropout=0):
        super(BDDVariableOrderingModel, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = GraphSAGEEncoder(in_feats, hidden_size, num_layers, dropout)

        # Pointer Decoder
        self.decoder = PointerDecoder(hidden_size)

    def forward(self, g, features, node_types, target_sequence=None, teacher_forcing_ratio=0.5):
        device = features.device

        # Encode the graph
        encoder_outputs = self.encoder(g, features)  # Shape: [total_num_nodes, hidden_size]

        # Prepare per-graph encoder outputs and mappings
        batch_num_nodes = g.batch_num_nodes().tolist()
        batch_size = len(batch_num_nodes)
        cumulative_num_nodes = torch.cumsum(torch.tensor([0] + batch_num_nodes[:-1]), dim=0)

        encoder_outputs_batch = []
        max_num_input_nodes = 0
        node_idx_to_decoder_idx_batch = []
        for i in range(batch_size):
            start_idx = cumulative_num_nodes[i]
            end_idx = cumulative_num_nodes[i] + batch_num_nodes[i]

            node_types_i = node_types[start_idx:end_idx]
            input_node_indices = torch.where(node_types_i == 0)[0]  # Indices within the graph

            if input_node_indices.numel() == 0:
                # Handle graphs with no input nodes
                input_node_indices = torch.tensor([], dtype=torch.long, device=device)
                encoder_outputs_i = torch.zeros(1, self.hidden_size, device=device)
            else:
                encoder_outputs_i = encoder_outputs[start_idx + input_node_indices]

            encoder_outputs_batch.append(encoder_outputs_i)
            if encoder_outputs_i.size(0) > max_num_input_nodes:
                max_num_input_nodes = encoder_outputs_i.size(0)

            # Create mapping from node indices to decoder indices
            node_idx_to_decoder_idx = {int(input_node_indices[j].item()): j for j in range(len(input_node_indices))}
            node_idx_to_decoder_idx_batch.append(node_idx_to_decoder_idx)

        # Pad encoder outputs to have the same number of input nodes
        for i in range(batch_size):
            encoder_output = encoder_outputs_batch[i]
            pad_size = max_num_input_nodes - encoder_output.size(0)
            if pad_size > 0:
                padding = torch.zeros(pad_size, self.hidden_size, device=device)
                encoder_output = torch.cat([encoder_output, padding], dim=0)
            encoder_outputs_batch[i] = encoder_output.unsqueeze(0)  # Shape: [1, max_num_input_nodes, hidden_size]

        # Stack encoder outputs
        encoder_outputs_batch = torch.cat(encoder_outputs_batch, dim=0)  # Shape: [batch_size, max_num_input_nodes, hidden_size]

        # Adjust target_sequence if available
        if target_sequence is not None:
            # Map node indices to decoder indices for each sample
            target_sequence_mapped = []
            for i in range(batch_size):
                node_idx_to_decoder_idx = node_idx_to_decoder_idx_batch[i]
                end_token_decoder_idx = len(node_idx_to_decoder_idx)
                node_idx_to_decoder_idx[-1] = end_token_decoder_idx  # Map end token

                target_seq = target_sequence[i]
                mapped_seq = []
                for node_idx in target_seq:
                    node_idx = int(node_idx)
                    if node_idx == -1:
                        mapped_seq.append(end_token_decoder_idx)
                    else:
                        mapped_seq.append(node_idx_to_decoder_idx.get(node_idx, end_token_decoder_idx))
                target_sequence_mapped.append(torch.tensor(mapped_seq, dtype=torch.long, device=device))

            # Pad target sequences
            target_sequence = torch.nn.utils.rnn.pad_sequence(target_sequence_mapped, batch_first=True, padding_value=-1)

        # Decode the sequence
        outputs, selected_indices = self.decoder(encoder_outputs_batch, target_sequence, teacher_forcing_ratio)

        return outputs, selected_indices
