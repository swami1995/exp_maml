import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import OrderedDict
import ipdb

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class RewardNetMLP(nn.Module):
    def __init__(self, state_size, action_size, embedding_size, separate_actions=False, output_size=1, hidden_sizes_pre_embedding=(),
                 hidden_sizes_post_embedding=(), nonlinearity=F.relu):
        super(RewardNetMLP, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.separate_actions = separate_actions
        if self.separate_actions:
            self.input_size = state_size+embedding_size
        else:
            self.input_size = state_size+action_size+embedding_size                        ### NOTE: Needs to be changed for different kinds of inputs
        self.embedding_size = embedding_size
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

        if self.separate_actions:
            layer_sizes = (self.action_size,) + hidden_sizes_pre_embedding
            for i in range(1, self.num_layers_pre):
                self.add_module('layer_pre_action{0}'.format(i),
                    nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

            layer_sizes = ((hidden_sizes_pre_embedding[-1],)#+self.embedding_size,) 
                            + hidden_sizes_post_embedding)
            for i in range(1, self.num_layers_post):
                self.add_module('layer_post_action{0}'.format(i+self.num_layers_pre-1),
                    nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

            self.reward_layer = nn.Linear(layer_sizes[-1]*2, output_size)
        else:
            self.reward_layer = nn.Linear(layer_sizes[-1], output_size)

        self.apply(weight_init)

    def set_hyperparams(self, use_successor_rep=False, num_step_returns=300, corrected_successor_rep=True, use_successor_rep_action=True):
        self.use_successor_rep=use_successor_rep
        self.num_step_returns = num_step_returns
        self.use_successor_rep_action = use_successor_rep_action
        self.corrected_successor_rep = corrected_successor_rep

    def update_params(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(),
            create_graph=not first_order)
        updated_params = OrderedDict()
        curr_params = self.named_parameters()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return curr_params, updated_params


    def forward(self, state, action, z, mask, params=None, ph=False):
        num_step_returns = min(self.num_step_returns, state.shape[0])
        new_z = z.unsqueeze(0).repeat(state.shape[0],state.shape[1],1)
        if ph:
            new_z_ph = new_z.detach()
            new_z_ph.requires_grad_()
            if self.separate_actions:
                input = torch.cat([state,new_z], dim=-1)
            else:
                input = torch.cat([state,action,new_z],dim=-1)
        else:
            if self.separate_actions:
                input = torch.cat([state,new_z], dim=-1)
            else:
                input = torch.cat([state,action,new_z],dim=-1)        
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers_pre):
            output = F.linear(output,
                weight=params['layer_pre{0}.weight'.format(i)],
                bias=params['layer_pre{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        # pdb.set_trace()
        # new_z = z.unsqueeze(0).repeat(output.shape[0],output.shape[1],1)
        # output = torch.cat([output,new_z], dim=-1)
        if self.use_successor_rep:
            if self.corrected_successor_rep:
                output = output*mask
                output_cumsum = torch.cat([output, torch.zeros_like(output[:num_step_returns])], dim=0).cumsum(0)/num_step_returns
                output = output_cumsum[num_step_returns:] - output_cumsum[:-num_step_returns]
            else:
                output_cumsum = output.cumsum(0)/num_step_returns
                output = output_cumsum - torch.cat([output_cumsum[num_step_returns:], torch.zeros_like(output_cumsum[:num_step_returns])], dim=0)

        for i in range(1, self.num_layers_post):
            output = F.linear(output,
                weight=params['layer_post{0}.weight'.format(i+self.num_layers_pre-1)],
                bias=params['layer_post{0}.bias'.format(i+self.num_layers_pre-1)])
            output = self.nonlinearity(output)  


        if self.separate_actions:
            output_action = action
            for i in range(1, self.num_layers_pre):
                output_action = F.linear(output_action,
                    weight=params['layer_pre_action{0}.weight'.format(i)],
                    bias=params['layer_pre_action{0}.bias'.format(i)])
                output_action = self.nonlinearity(output_action)

            # pdb.set_trace()
            # new_z = z.unsqueeze(0).repeat(output.shape[0],output.shape[1],1)
            # output = torch.cat([output,new_z], dim=-1)
            if self.use_successor_rep_action:
                output_action = output_action*mask
                output_cumsum = torch.cat([output_action, torch.zeros_like(output_action[:num_step_returns])], dim=0).cumsum(0)/num_step_returns
                output_action = output_cumsum[num_step_returns:] - output_cumsum[:-num_step_returns]

            for i in range(1, self.num_layers_post):
                output_action = F.linear(output_action,
                    weight=params['layer_post_action{0}.weight'.format(i+self.num_layers_pre-1)],
                    bias=params['layer_post_action{0}.bias'.format(i+self.num_layers_pre-1)])
                output_action = self.nonlinearity(output_action)
            output = torch.cat([output, output_action], dim=-1)  
        reward = F.linear(output, weight=params['reward_layer.weight'],
            bias=params['reward_layer.bias'])
        if ph:
            return reward, new_z
        else:
            return reward


class RewardNetMLP_shared(nn.Module):
    def __init__(self, state_size, action_size, embedding_size, output_size=1, hidden_sizes=(),
                 nonlinearity=F.relu):
        super(RewardNetMLP_shared, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = state_size+action_size                        ### NOTE: Needs to be changed for different kinds of inputs
        self.embedding_size = embedding_size
        self.output_size = output_size

        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 2

        layer_sizes = (self.input_size,) + hidden_sizes  + (self.embedding_size,)
        for i in range(1, self.num_layers):
            self.add_module('layer_pre{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.bias_out = nn.Parameter(torch.zeros(1))

        self.apply(weight_init)

    def forward(self, state, action, z, params=None, ph=False):
        if ph:
            z_ph = z.detach()
            z_ph.requires_grad_()
        input = torch.cat([state,action],dim=-1)
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer_pre{0}.weight'.format(i)],
                bias=params['layer_pre{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        reward = F.linear(output, weight=z,
            bias=self.bias_out)
        if ph:
            return reward, z
        else:
            return reward