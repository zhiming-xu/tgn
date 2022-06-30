import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False, 
                 mem_node_prob=1, use_fixed_times=False):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]    # xzl) num of nodes
        self.n_edge_features = self.edge_raw_features.shape[1]  # xzl) edge feat dimension?
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None
        self.mem_node_prob = mem_node_prob

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            # xzl: whole graph memory, for each node
            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             device=device)
            self.message_function = get_message_function(module_type=message_function,
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            # xzl: GRU/RNN... only has weights, no memory inside
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)

        self.embedding_module_type = embedding_module_type

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_raw_features,
                                                     edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_node_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     use_fixed_times=use_fixed_times,
                                                     n_neighbors=self.n_neighbors)

        # MLP to compute probability on an edge given two node embeddings
        # xzl: decoder, MLP. traineable model.
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction xzl) used to index edge features
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives

        xzl: the reason compute pos/neg together -- batch them in one shot. 
        afte computing embeddings, update memory 
        """

        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes]) #xzl: all nodes
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])     # xzl: ts for an edge... for src/dest/neg(dest) ... ?

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches 
                # xzl: pull msgs of last batch, cal @memory which is used to cal emeddings 
                #       only after embeddings are cal, cal @memory again and persist it
                #     @self.memory.messages is from previous batch        
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                              self.memory.messages)
            else:
                # xzl: memory already updated at the end of last batch. now just retrieve it
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            ### Compute differences between the time the memory of a node was last updated,
            ### and the time for which we want to compute the embedding of a node
            # xzl: for encoding the time. normalize times by mean/std...
            #       only used for TimeEmbedding (Jodie?). not by GraphSum and GAT embeddings, which uses @timestamps
            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                negative_nodes].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                                   dim=0)

        # Compute the embeddings using the embedding module
        # xzl: @nodes has all nodes (src,dest,dest-neg). @timestamps is for time embeddings
        #print("------------------ xzl: timestamps size", timestamps.shape)
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

        if self.use_memory:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)      xzl: redundant computation?
                self.update_memory(positives, self.memory.messages)

                # xzl: online discussion said it's okay to ignore below
                #assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
                #  "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)

            # xzl) fetch current batch's msgs only after cal current embeddings..., store to self.memory
            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                          source_node_embedding,
                                                                          destination_nodes,
                                                                          destination_node_embedding,
                                                                          edge_times, edge_idxs)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                    destination_node_embedding,
                                                                                    source_nodes,
                                                                                    source_node_embedding,
                                                                                    edge_times, edge_idxs)
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                negative_node_embedding = memory[negative_nodes]

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def compute_embedding_and_update_memory(self, source_nodes, destination_nodes, edge_times,
                                            edge_idxs, n_neighbors=20):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction xzl) used to index edge features
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives

        xzl: the reason compute pos/neg together -- batch them in one shot. 
        afte computing embeddings, update memory 
        """

        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes]) #xzl: all nodes
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])     # xzl: ts for an edge... for src/dest/neg(dest) ... ?

        memory = None
        time_diffs = None
        if self.use_memory:
            # xzl: memory already updated at the end of last batch. now just retrieve it
            memory = self.memory.get_memory(list(range(self.n_nodes)))
            last_update = self.memory.last_update

        # Compute the embeddings using the embedding module
        # xzl: @nodes has all nodes (src,dest,dest-neg). @timestamps is for time embeddings
        #print("------------------ xzl: timestamps size", timestamps.shape)
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

        if self.use_memory:
            # xzl) fetch current batch's msgs only after cal current embeddings..., store to self.memory
            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                          source_node_embedding,
                                                                          destination_nodes,
                                                                          destination_node_embedding,
                                                                          edge_times, edge_idxs)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                    destination_node_embedding,
                                                                                    source_nodes,
                                                                                    source_node_embedding,
                                                                                    edge_times, edge_idxs)
            self.update_memory(unique_sources, source_id_to_messages)
            self.update_memory(unique_destinations, destination_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
        # no need to return, just update memory
        # return source_node_embedding, destination_node_embedding

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, n_neighbors=20):
        """
        Compute probabilities for edges between sources and destination and between sources and
        negatives by first computing temporal embeddings using the TGN encoder and then feeding them
        into the MLP decoder.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Probabilities for both the positive and negative edges
        """
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                               negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def compute_all_pairs_probabilities(self, source_node, edge_time, candidates, n_neighbors):
        n_samples = len(candidates)
        nodes = np.concatenate([[source_node], candidates]) #xzl: all nodes
        timestamps = np.array([edge_time] * nodes.size)     # xzl: ts for an edge... for src/dest/neg(dest) ... ?

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches 
                # xzl: pull msgs of last batch, cal @memory which is used to cal emeddings 
                #       only after embeddings are cal, cal @memory again and persist it
                #     @self.memory.messages is from previous batch        
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                              self.memory.messages)
            else:
                # xzl: memory already updated at the end of last batch. now just retrieve it
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            ### Compute differences between the time the memory of a node was last updated,
            ### and the time for which we want to compute the embedding of a node
            # xzl: for encoding the time. normalize times by mean/std...
            #       only used for TimeEmbedding (Jodie?). not by GraphSum and GAT embeddings, which uses @timestamps

        # Compute the embeddings using the embedding module
        # xzl: @nodes has all nodes (src,dest,dest-neg). @timestamps is for time embeddings
        #print("------------------ xzl: timestamps size", timestamps.shape)
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[0]
        destination_node_embedding = node_embedding[1:]
        '''
    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)      xzl: redundant computation?
        self.update_memory(positives, self.memory.messages)

        # xzl: online discussion said it's okay to ignore below
        #assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
        #  "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      # xzl) fetch current batch's msgs only after cal current embeddings..., store to self.memory
      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]
    '''
        source_node_embedding = source_node_embedding.repeat(n_samples, 1)
        score = self.affinity_score(source_node_embedding, destination_node_embedding)
        return score.sigmoid()

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        # xzl) first agg then msg func ... seems intentional
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
            messages)

        # xzl sampling nodes...
        # print(f"xzl: before sampling. #unique_nodes {len(unique_nodes)}")    
        if self.mem_node_prob < 0.9999:
            idx = np.random.choice(np.arange(len(unique_nodes)), 
              int(len(unique_nodes)*self.mem_node_prob), replace=False)
            sampled_nodes = np.array(unique_nodes)[idx] 
            #     using sampled nodes, agg message again 
            unique_nodes, unique_messages, unique_timestamps = \
              self.message_aggregator.aggregate(
                  sampled_nodes,
                messages)

        # xzl sample --- works, but there's a better way above.
        #print(len(unique_nodes), type(unique_nodes), type(unique_messages))
        # if len(unique_nodes) > 0:
        #   p = 0.5
        #   idx = np.random.choice(np.arange(len(unique_nodes)), len(unique_nodes)>>1, replace=False)
        #   #print(len(unique_nodes), idx, type(idx), type(unique_nodes), type(unique_messages))
        #   unique_nodes = np.array(unique_nodes)[idx] 
        #   unique_messages = torch.from_numpy(np.array(unique_messages.cpu())[idx]).to('cuda:0')
        #   unique_timestamps =torch.from_numpy(np.array(unique_timestamps.cpu())[idx]).to('cuda:0')

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # print(f"\t\t xzl: after sampling. #unique_nodes {len(unique_nodes)}")
        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                         timestamps=unique_timestamps)

        # xzl: sample...  
        # if len(unique_nodes) > 0:
        #   p = 0.5
        #   idx = np.random.choice(np.arange(len(unique_nodes)), len(unique_nodes)>>1, replace=False)
        #   #print(len(unique_nodes), idx, type(idx), type(unique_nodes), type(unique_messages))
        #   self.memory_updater.update_memory(np.array(unique_nodes)[idx], 
        #   np.array(unique_messages)[idx],
        #                                     timestamps=np.array(unique_timestamps)[idx])

    # xzl) to be called prior to embedding update. 
    #       cal updated mem per last batch messages (?). not updating the memory state
    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
            messages)

        # xzl: sampling nodes TODO -- deterministic sampling
        if self.mem_node_prob < 0.9999:
            idx = np.random.choice(np.arange(len(unique_nodes)), 
              int(len(unique_nodes)*self.mem_node_prob), replace=False)
            sampled_nodes = np.array(unique_nodes)[idx] 
            #     using sampled nodes, agg message again 
            unique_nodes, unique_messages, unique_timestamps = \
              self.message_aggregator.aggregate(
                  sampled_nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    # xzl: given "edges", return a dict: source_nodes->((msg1,t1),(msg2,t2)...), 
    # "raw msg" because each @msg carries lots of info, which is yet to send thorugh the msg func
    # NB: can be invoked w/ @source_nodes/@dest_nodes swapped. therefore can return mappings from 
    # dest nodes -> msgs
    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, edge_times, edge_idxs):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]

        source_memory = self.memory.get_memory(source_nodes) if not \
          self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
          not self.use_destination_embedding_in_message else destination_node_embedding

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
            source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    source_time_delta_encoding],
                                   dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        # xzl) below expensive...
        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        # xzl: @messages is a dict, source_node -> list(m1,m2,m3...)
        # dest_node not saved??
        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
