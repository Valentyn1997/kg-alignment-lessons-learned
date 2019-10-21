# coding=utf-8
import logging
import math
from types import FunctionType

import torch
from torch import nn
from torch.nn import functional, init

from modules.common import EdgeWeightsEnum, calculate_inverse_in_degree_edge_weights
from utils.common import truncated_normal_


class GCNAlign(nn.Module):
    def __init__(
        self,
        num_match_nodes: int,
        num_ref_nodes: int,
        match_edge_tensor: torch.LongTensor,
        ref_edge_tensor: torch.LongTensor,
        embedding_dim: int = 200,
        device: torch.device = torch.device('cpu'),
        activation_cls: nn.Module = nn.ReLU,
        n_layers: int = 2,
        use_edge_weights: EdgeWeightsEnum = EdgeWeightsEnum.inverse_in_degree,
        use_conv_weights: bool = True,
        conv_weight_init: FunctionType = init.xavier_uniform_,
        node_embedding_init: str = 'total',  # 'individual'
        dropout: float = 0.,
        vertical_sharing: bool = True,
        *args, **kwargs
    ):
        super().__init__()
        old_size = [t.shape[1] for t in (match_edge_tensor, ref_edge_tensor)]
        match_edge_tensor = torch.unique(match_edge_tensor, dim=1)
        ref_edge_tensor = torch.unique(ref_edge_tensor, dim=1)
        new_size = [t.shape[1] for t in (match_edge_tensor, ref_edge_tensor)]
        logging.info(f'Aggregated edges: {old_size} -> {new_size}.')

        if use_edge_weights == EdgeWeightsEnum.none:
            logging.info('Using uniform edge weights.')
            self.match_edge_weights = None
            self.ref_edge_weights = None
        elif use_edge_weights == EdgeWeightsEnum.inverse_in_degree:
            logging.info('Using inverse in-degree edge weights.')
            match_edge_weights = calculate_inverse_in_degree_edge_weights(match_edge_tensor)
            self.register_buffer(name='match_edge_weights', tensor=match_edge_weights.to(device=device))

            ref_edge_weights = calculate_inverse_in_degree_edge_weights(ref_edge_tensor)
            self.register_buffer(name='ref_edge_weights', tensor=ref_edge_weights.to(device=device))
        else:
            raise KeyError(use_edge_weights)

        self.register_buffer(name='match_edge_tensor', tensor=match_edge_tensor.to(device=device))
        self.register_buffer(name='ref_edge_tensor', tensor=ref_edge_tensor.to(device=device))
        self.match_node_embeddings = nn.Parameter(torch.empty(num_match_nodes, embedding_dim, device=device))
        self.ref_node_embeddings = nn.Parameter(torch.empty(num_ref_nodes, embedding_dim, device=device))
        self.node_embedding_init = node_embedding_init

        self.n_layers = n_layers
        self.use_conv_weights = use_conv_weights
        # for _ in range(self.n_layers):
        self.match_weights = None
        self.vertical_sharing = vertical_sharing
        self.match_biases = 0
        if self.use_conv_weights:
            if self.vertical_sharing:
                self.match_weights = nn.Parameter(torch.empty(embedding_dim, embedding_dim, device=device))
                self.match_biases = nn.Parameter(torch.zeros(embedding_dim, device=device))
            else:
                self.match_weights = [nn.Parameter(torch.empty(embedding_dim, embedding_dim, device=device)) for _ in range(self.n_layers)]
                self.match_biases = [nn.Parameter(torch.zeros(embedding_dim, device=device)) for _ in range(self.n_layers)]
            self.conv_weight_init = conv_weight_init
        self.ref_weights = self.match_weights  # nn.Parameter(torch.randn(embedding_dim, embedding_dim, device=device))
        self.ref_biases = self.match_biases
        # self.match_weights.append(match_weight)
        # self.ref_weights.append(ref_weight)

        self.act = activation_cls()
        self.dropout = dropout
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        if self.node_embedding_init == 'total':
            total_num_nodes = self.match_node_embeddings.shape[0] + self.ref_node_embeddings.shape[0]
            truncated_normal_(self.match_node_embeddings, std=1 / math.sqrt(total_num_nodes))
            truncated_normal_(self.ref_node_embeddings, std=1 / math.sqrt(total_num_nodes))
        elif self.node_embedding_init == 'individual':
            truncated_normal_(self.match_node_embeddings, std=1 / math.sqrt(self.match_node_embeddings.shape[0]))
            truncated_normal_(self.ref_node_embeddings, std=1 / math.sqrt(self.ref_node_embeddings.shape[0]))
        elif self.node_embedding_init == 'none':
            truncated_normal_(self.match_node_embeddings, std=1)
            truncated_normal_(self.ref_node_embeddings, std=1)
        if self.use_conv_weights:
            if self.vertical_sharing:
                self.conv_weight_init(self.match_weights)
                init.zeros_(self.match_biases)
            else:
                for match_weight, match_bias in zip(self.match_weights, self.match_biases):
                    self.conv_weight_init(match_weight)
                    init.zeros_(match_bias)



    def forward(self):
        result = []
        for node_emb, edge_tensor, weights, biases, edge_weights in zip(
            [self.match_node_embeddings, self.ref_node_embeddings],
            [self.match_edge_tensor, self.ref_edge_tensor],
            [self.match_weights, self.ref_weights],
            [self.match_biases, self.ref_biases],
            [self.match_edge_weights, self.ref_edge_weights],
        ):
            # cf. https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/models.py#L205-L214
            # cf. https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/inits.py#L31-L34
            node_emb = functional.normalize(node_emb, p=2, dim=1)

            for layer in range(self.n_layers):
                # Send messages without weight
                hidden = self.send_messages(edge_tensor=edge_tensor, source_data=node_emb, edge_weights=edge_weights)

                # Multiply on weight matrix (vertically shared)
                if self.use_conv_weights:
                    if self.vertical_sharing:
                        hidden = hidden @ weights + biases
                    else:
                        hidden = hidden @ weights[layer] + biases[layer]

                # Activation
                node_emb = self.act(hidden)

            result.append(node_emb)
        return result

    def send_messages(self, edge_tensor, source_data, edge_weights=None):
        # Send messages to edges
        source, target = edge_tensor
        msg = torch.index_select(source_data, dim=0, index=source)

        if edge_weights is not None:
            msg = msg * edge_weights

        # Accumulate messages
        acc = torch.zeros_like(source_data)
        out = torch.index_add(acc, dim=0, index=target, source=msg)
        return out
