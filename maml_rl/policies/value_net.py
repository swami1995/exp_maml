import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import OrderedDict
import pdb

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class ValueNetMLP(nn.Module):
    def __init__(self, state_size, output_size=1, hidden_sizes_pre_embedding=(),
                 hidden_sizes_post_embedding=(), nonlinearity=F.relu):
        super().__init__()
        self.state_size = state_size
        self.input_size = state_size                       ### NOTE: Needs to be changed for different kinds of inputs
        self.output_size = output_size

        self.hidden_sizes_pre_embedding = hidden_sizes_pre_embedding
        self.nonlinearity = nonlinearity
        self.num_layers_pre = len(hidden_sizes_pre_embedding) + 1
        self.num_layers_post = len(hidden_sizes_post_embedding) + 1

        layer_sizes = (self.input_size,) + hidden_sizes_pre_embedding
        for i in range(1, self.num_layers_pre):
            self.add_module('layer_pre{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        layer_sizes = ((hidden_sizes_pre_embedding[-1],)#+self.embedding_size,) 
                        + hidden_sizes_post_embedding)
        for i in range(1, self.num_layers_post):
            self.add_module('layer_post{0}'.format(i+self.num_layers_pre-1),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.last_layer = nn.Linear(layer_sizes[-1], output_size)

        self.apply(weight_init)

    def forward(self, state, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = state
        for i in range(1, self.num_layers_pre):
            output = F.linear(output,
                weight=params['layer_pre{0}.weight'.format(i)],
                bias=params['layer_pre{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        for i in range(1, self.num_layers_post):
            output = F.linear(output,
                weight=params['layer_post{0}.weight'.format(i+self.num_layers_pre-1)],
                bias=params['layer_post{0}.bias'.format(i+self.num_layers_pre-1)])
            output = self.nonlinearity(output)  

        Value = F.linear(output, weight=params['last_layer.weight'],
            bias=params['last_layer.bias'])
        return Value