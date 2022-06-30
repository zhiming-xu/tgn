import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer

# xzl: code well structured...!!

class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features, n_time_features,
                                            embedding_dimension, device, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

        return source_embeddings

# xzl: embedding considering neighbohood. superclass of GraphSum and GAT
class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, use_fixed_times=False):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device
        self.use_fixed_times = use_fixed_times

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.   (xzl: source_nodes
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        (xzl: ^ timestamps. used to cal time embeddings (source and edge(diff))

        curr_layers [scalar]: number of temporal convolutional layers to stack. (xzl: ^ n_layers
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        # xzl: use same timestamps... for all
        if self.use_fixed_times: 
            timestamps.fill(np.mean(timestamps))

        # xzl: assemble tensors to upload to GPU
        # print("xzl: timestamps for the batch", timestamps)    #xzl: still absolute values.
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # print("------------------ xzl: timestamps_torch size", timestamps_torch.size())

        # query node always has the start time -> time span == 0
        #    xzl:  "query node"  -- the GAT notation (source node)
        #     time encodings: for the "query node", then for edges (within 1-hop temp neighbors)
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        # xzl) layer 0 input: node feature + memory
        if n_layers == 0:
            return source_node_features
        else:

            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
              timestamps,
              n_neighbors=n_neighbors)

            # xzl: use same @edge_times for all ... no effect on accuracy...
            if self.use_fixed_times: 
                edge_times.fill(np.mean(edge_times))

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis] - edge_times # xzl) current time - edge time
            #print(edge_deltas)   # xzl: check if fixed ts work...

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            # xzl: recurse to compute next hop neighbord embeddings..., L-1
            neighbor_embeddings = self.compute_embedding(memory,
                                                         neighbors,
                                                         np.repeat(timestamps, n_neighbors),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors_torch == 0 # xzl) for padded neighbors

            # xzl: agg from: neighbor embeddings, edge time (elapsed), source time (ts)--> source embeddings
            source_embedding = self.aggregate(n_layers, source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

            return source_embedding

    # xzl: @aggregate can overriden by subclass
    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        return None


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, use_fixed_times=False):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                edge_features=edge_features,
                                                memory=memory,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder, n_layers=n_layers,
                                                n_node_features=n_node_features,
                                                n_edge_features=n_edge_features,
                                                n_time_features=n_time_features,
                                                embedding_dimension=embedding_dimension,
                                                device=device,
                                                n_heads=n_heads, dropout=dropout,
                                                use_memory=use_memory, use_fixed_times=use_fixed_times)
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                             n_edge_features, embedding_dimension)
                                             for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                           embedding_dimension) for _ in range(n_layers)])

    #xzl) caller already adds memory to @source_node_features 
    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        # xzl: cat neighbors, sum, relu. then cat sources || neighbors, linear
        # @dim: the dimension over which the tensors are concatenated
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                       dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, use_fixed_times=False):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory, use_fixed_times)

        # xzl: put together @n_layers of attn layers...
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
          n_neighbors_features=n_node_features,
          n_edge_features=n_edge_features,
          time_dim=n_time_features,
          n_head=n_heads,
          dropout=dropout,
          output_dimension=n_node_features)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

        return source_embedding

# xzl) a global func
def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True, use_fixed_times=False):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                        edge_features=edge_features,
                                        memory=memory,
                                        neighbor_finder=neighbor_finder,
                                        time_encoder=time_encoder,
                                        n_layers=n_layers,
                                        n_node_features=n_node_features,
                                        n_edge_features=n_edge_features,
                                        n_time_features=n_time_features,
                                        embedding_dimension=embedding_dimension,
                                        device=device,
                                        n_heads=n_heads, dropout=dropout, use_memory=use_memory, 
                                        use_fixed_times=use_fixed_times)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                  edge_features=edge_features,
                                  memory=memory,
                                  neighbor_finder=neighbor_finder,
                                  time_encoder=time_encoder,
                                  n_layers=n_layers,
                                  n_node_features=n_node_features,
                                  n_edge_features=n_edge_features,
                                  n_time_features=n_time_features,
                                  embedding_dimension=embedding_dimension,
                                  device=device,
                                  n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                  use_fixed_times=use_fixed_times)

    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features,
                                 edge_features=edge_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout,
                             n_neighbors=n_neighbors)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))


