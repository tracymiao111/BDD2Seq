# model_ptr_test.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, in_feats, hidden_size, num_heads, num_layers, dropout):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Input layer
        self.layers.append(GATConv(
            in_feats, hidden_size, num_heads,
            feat_drop=dropout, attn_drop=dropout,
            activation=F.elu, allow_zero_in_degree=True
        ))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(
                hidden_size * num_heads, hidden_size, num_heads,
                feat_drop=dropout, attn_drop=dropout,
                activation=F.elu, allow_zero_in_degree=True
            ))

        # Output layer
        self.layers.append(GATConv(
            hidden_size * num_heads, hidden_size, 1,  # Single head for output layer
            feat_drop=dropout, attn_drop=dropout,
            activation=None, allow_zero_in_degree=True
        ))

    def forward(self, g, features):
        h = features
        for l in range(self.num_layers - 1):
            h = self.layers[l](g, h).flatten(1)  # Flatten the heads
        # Output layer
        h = self.layers[-1](g, h)  # Shape: [num_nodes, hidden_size, num_heads]
        h = h.mean(dim=1)  # Average over heads if needed
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
        extended_encoder_outputs = torch.cat([encoder_outputs, end_token_embedding], dim=1)

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
            hx, cx = self.decoder_cell(decoder_input, (hx, cx))

            # Attention mechanism
            attn_scores = self.attn_v(torch.tanh(
                self.attn_W_a(extended_encoder_outputs) + self.attn_U_a(hx).unsqueeze(1)
            )).squeeze(-1)

            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
            attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]
            log_probs = F.log_softmax(attn_scores, dim=1)

            # Teacher forcing vs. model argmax
            if target_sequence is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_idx = target_sequence[:, t]
            else:
                next_idx = log_probs.argmax(dim=1)

            mask_indices = next_idx < num_input_nodes
            batch_indices = torch.arange(batch_size, device=device)
            mask[batch_indices[mask_indices], next_idx[mask_indices]] = 1

            decoder_input = torch.zeros(batch_size, hidden_size, device=device)
            for b in range(batch_size):
                idx = next_idx[b]
                if idx < num_input_nodes:
                    decoder_input[b] = encoder_outputs[b, idx]
                else:
                    decoder_input[b] = self.end_token

            outputs.append(log_probs)
            selected_indices.append(next_idx)

        outputs = torch.stack(outputs, dim=1)
        selected_indices = torch.stack(selected_indices, dim=1)
        return outputs, selected_indices

    ######################################################################
    # MODIFIED diverse_beam_search BELOW
    ######################################################################
    def diverse_beam_search(
        self,
        encoder_outputs,
        beam_width=5,
        num_groups=5,
        diversity_strength=0.5,
        max_decoding_steps=None,
        max_sequences=100,    # <-- NEW: up to 100 distinct sequences
    ):
        """
        Modified diverse beam search that attempts to produce up to 'max_sequences'
        unique final sequences. If fewer can be found, it returns as many as possible.

        :param encoder_outputs: [1, num_input_nodes, hidden_size]
        :param beam_width: total beam width
        :param num_groups: how many sub-beams for diversity
        :param diversity_strength: how strongly we penalize repeated tokens
        :param max_decoding_steps: optional override for max steps
        :param max_sequences: the maximum number of unique final sequences to return

        :return: A list of up to 'max_sequences' unique final sequences in
                 descending order of score (best first).
        """
        device = encoder_outputs.device
        batch_size, num_input_nodes, hidden_size = encoder_outputs.size()
        assert batch_size == 1, "diverse_beam_search supports batch_size=1."

        if max_decoding_steps is None:
            max_decoding_steps = num_input_nodes

        # Each beam: (score, decoder_input, hx, cx, seq, used_mask)
        # Start token
        start_input = self.start_token.unsqueeze(0).to(device)  # shape [1, hidden_size]
        hx = torch.zeros(1, hidden_size, device=device)
        cx = torch.zeros(1, hidden_size, device=device)

        # Initialize single beam, then replicate for each group
        initial_beam = (
            0.0,
            start_input,  # decoder_input
            hx, cx,
            [],  # seq
            torch.zeros(1, num_input_nodes, device=device),  # used_mask
        )
        group_size = beam_width // num_groups
        beams = [initial_beam] * num_groups * group_size

        completed_sequences = []      # store (score, sequence)
        completed_set = set()         # to filter duplicates

        # We'll store partial sequences for each group to apply diversity penalty
        group_sequences = [[] for _ in range(num_groups)]

        for step in range(max_decoding_steps):
            # Stop if we have enough distinct final sequences
            if len(completed_sequences) >= max_sequences:
                break

            new_beams = [[] for _ in range(num_groups)]  # store candidates per group

            # Process each group
            for g_idx in range(num_groups):
                # Slice out the beams for this group
                start_i = g_idx * group_size
                end_i = start_i + group_size
                group_beam_slice = beams[start_i:end_i]

                group_seq_cache = []

                # Expand each beam in this group
                for beam in group_beam_slice:
                    score, dec_in, hx, cx, seq, used_mask = beam

                    # LSTM step
                    hx, cx = self.decoder_cell(dec_in, (hx, cx))

                    # Attention
                    attn_scores = self.attn_v(torch.tanh(
                        self.attn_W_a(encoder_outputs) + self.attn_U_a(hx).unsqueeze(1)
                    )).squeeze(-1)  # shape: [1, num_input_nodes]

                    # Mask out used tokens
                    attn_scores = attn_scores.masked_fill(used_mask == 1, float('-inf'))

                    # Diversity penalty: penalize tokens used in other groups
                    # i.e., step-level penalty from prior groups
                    for prev_g_idx in range(g_idx):
                        for prev_seq in group_sequences[prev_g_idx]:
                            for idx_token in prev_seq:
                                # attn_scores[0, idx_token] -= diversity_strength
                                attn_scores[0, idx_token] *= (1 - diversity_strength)

                    # Compute log_probs
                    attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]
                    log_probs = F.log_softmax(attn_scores, dim=1)

                    # Count how many tokens remain valid
                    valid_mask = (used_mask == 0)
                    valid_count = valid_mask.sum().item()
                    if valid_count == 0:
                        # All tokens used => we can finalize this sequence
                        # But only if it's not a duplicate
                        t_seq = tuple(seq)
                        if t_seq not in completed_set:
                            completed_set.add(t_seq)
                            completed_sequences.append((score, seq))
                        continue

                    # Expand topK among the valid
                    k = min(group_size, valid_count)
                    topk_vals, topk_inds = log_probs.topk(k, dim=1)

                    for i in range(k):
                        next_idx = topk_inds[0, i].item()
                        next_lp = topk_vals[0, i].item()
                        new_score = score + next_lp

                        new_seq = seq + [next_idx]
                        new_mask = used_mask.clone()
                        new_mask[0, next_idx] = 1

                        # Next decoder input
                        next_dec_in = encoder_outputs[:, next_idx, :]

                        # If we've used all nodes => finalize
                        if len(new_seq) == num_input_nodes:
                            t_seq = tuple(new_seq)
                            if t_seq not in completed_set:
                                completed_set.add(t_seq)
                                completed_sequences.append((new_score, new_seq))
                        else:
                            # partial beam
                            new_beams[g_idx].append((
                                new_score, next_dec_in, hx.clone(), cx.clone(),
                                new_seq, new_mask
                            ))

                # We'll gather new_beams for the group. Sort & keep top group_size
                new_beams[g_idx].sort(key=lambda x: x[0], reverse=True)
                new_beams[g_idx] = new_beams[g_idx][:group_size]

                # For diversity penalty next iteration, store the seqs
                group_sequences[g_idx] = [b[4] for b in new_beams[g_idx]]

            # Flatten back into 'beams' list
            flattened = []
            for g_idx in range(num_groups):
                flattened.extend(new_beams[g_idx])
            beams = flattened

            # If no beams remain and we haven't reached max_sequences, we break
            if len(beams) == 0 and len(completed_sequences) < max_sequences:
                break

        # If we never got completed sequences, fallback to partial beams
        if not completed_sequences:
            for beam in beams:
                scr, _, _, _, sq, _ = beam
                t_sq = tuple(sq)
                if t_sq not in completed_set:
                    completed_set.add(t_sq)
                    completed_sequences.append((scr, sq))

        # Sort final completed sequences by descending score
        completed_sequences.sort(key=lambda x: x[0], reverse=True)

        # Slice up to max_sequences
        completed_sequences = completed_sequences[:max_sequences]

        # Convert to torch tensors
        final_seqs = []
        for (scr, seq) in completed_sequences:
            final_seqs.append(torch.tensor(seq, dtype=torch.long, device=device))

        return final_seqs
    ######################################################################
    # END MODIFICATIONS
    ######################################################################

class BDDVariableOrderingModel(nn.Module):
    def __init__(self, in_feats, hidden_size=512, num_heads=8, num_layers=6, dropout=0):
        super(BDDVariableOrderingModel, self).__init__()
        self.hidden_size = hidden_size

        # GAT Encoder
        self.encoder = GATEncoder(in_feats, hidden_size, num_heads, num_layers, dropout)

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
