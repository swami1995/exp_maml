import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init
import pdb

class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """
    def __init__(self, input_size, output_size, embedding_size, hidden_sizes_pre_embedding=(),
                 hidden_sizes_post_embedding=(), nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size, embedding_size=embedding_size)

        self.hidden_sizes_pre_embedding = hidden_sizes_pre_embedding
        self.hidden_sizes_post_embedding = hidden_sizes_post_embedding
        self.nonlinearity = nonlinearity
        self.num_layers_pre = len(hidden_sizes_pre_embedding)
        self.num_layers_post = len(hidden_sizes_post_embedding)
        self.min_log_std = math.log(min_std)

        layer_sizes = (self.input_size,) + hidden_sizes_pre_embedding
        for i in range(1, self.num_layers_pre+1):
            self.add_module('layer_pre{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        
        layer_sizes = (hidden_sizes_pre_embedding[-1]+self.embedding_size,) + hidden_sizes_post_embedding
        for i in range(1, self.num_layers_post+1):
            self.add_module('layer_post{0}'.format(i+self.num_layers_pre),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))

        self.Value_fn = nn.Linear(layer_sizes[-1], 1)

        self.apply(weight_init)

    def forward(self, input, z, params=None):
        batch_shape = input.shape[:-1]
        input = input.reshape((-1,input.shape[-1]))
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers_pre+1):
            output = F.linear(output,
                weight=params['layer_pre{0}.weight'.format(i)],
                bias=params['layer_pre{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        output = torch.cat([output,z.repeat(output.shape[0],1)], dim=-1)
        for i in range(1, self.num_layers_post+1):
            output = F.linear(output,
                weight=params['layer_post{0}.weight'.format(i+self.num_layers_pre)],
                bias=params['layer_post{0}.bias'.format(i+self.num_layers_pre)])
            output = self.nonlinearity(output)  
        # pdb.set_trace()
        output = output.reshape(batch_shape+(-1,))
        mu = F.linear(output, weight=params['mu.weight'],
            bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        value = self.Value_fn(output)

        return Normal(loc=mu, scale=scale), value