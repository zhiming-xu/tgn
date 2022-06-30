import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy

# xzl: 
# holding: 
#   a global object containing memory for all nodes; 
#   messages (a dict, dest node -> messages)
# wraps around nn.Parameters (self.memory)
# 

class Memory(nn.Module):

    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
                 device="cpu", combination_method='sum'):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device

        self.combination_method = combination_method

        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                                   requires_grad=False)
        # xzl: last update time (per node)??                              
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                        requires_grad=False)

        # xzl: the msgs are only for the prev batch.
        self.messages = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    # xzl: a snapshot of memory... and raw messages?
    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach_() # xzl) no backprop at this point, but self.memory has requires_grad=False already??

        # Detach all stored messages   xzl) why not detach_() in place? immutable?
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []

    # xzl: clear all state, last batch msgs
    def clear_memory(self): 
        self.memory.data.zero_()
        self.last_update.data.zero_()
        for k, v in self.messages.items():
            self.messages[k] = []