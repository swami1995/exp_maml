import torch
from torch.distributions import Categorical, Normal
import ipdb
def weighted_mean(tensor, dim=None, weights=None):
    if weights is None:
        out = torch.mean(tensor)
        return out
    if dim is None:
        out = torch.sum(tensor * weights)
        out.div_(torch.sum(weights))
    else:
        mean_dim = torch.sum(tensor * weights, dim=dim)
        mean_dim.div_(torch.sum(weights, dim=dim))
        out = torch.mean(mean_dim)
    return out

def weighted_normalize(tensor, dim=None, weights=None, epsilon=1e-8, no_mean=False):
    mean = weighted_mean(tensor, dim=dim, weights=weights)
    out = tensor * (1 if weights is None else weights)
    if not no_mean:
        out-=mean
    std = torch.sqrt(weighted_mean(out ** 2, dim=dim, weights=weights))
    out.div_(std + epsilon)
    return out

def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution

def moving_weighted_normalize(tensor, dim=None, weights=None, epsilon=1e-8, no_mean=False, moving_params=None, alpha=0.01):
    mean = weighted_mean(tensor, dim=dim, weights=weights)
    out = tensor * (1 if weights is None else weights)
    std = torch.sqrt(weighted_mean(out ** 2, dim=dim, weights=weights))
    if moving_params is not None:
        # ipdb.set_trace()
        if moving_params[1]==1.:
            moving_params = torch.stack([mean, std])
        else:
            moving_params = moving_params + alpha * (torch.stack([mean, std]) - moving_params)
    else:
        moving_params = torch.stack([mean, std])
    if not no_mean:
        out-=moving_params[0]
    out.div_(moving_params[1] + epsilon)
    if moving_params is None:
        return out
    return out, moving_params