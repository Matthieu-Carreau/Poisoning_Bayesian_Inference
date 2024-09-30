import torch
from torch.distributions.kl import kl_divergence


def remove_zeros(tensor, eps=1e-10):
    """Replace the non positive values in a tensor by a small epsilon"""
    ret = tensor.clone().float()
    ret[ret <= 0] = eps
    return ret

def to_np(tensor):
    """Convert a torch tensor to a numpy array"""
    return tensor.detach().cpu().numpy()

def to_tensor(lst):
    """
    Convert a list of torch tensors or numpy arrays 
    to a single torch tensor
    """
    element_shape = lst[0].shape if hasattr(lst[0], "shape") else ()
    shape = (len(lst), *element_shape)
    
    if isinstance(lst[0], torch.Tensor):
        dtype = lst[0].dtype
    else:
        dtype = torch.tensor(lst[0]).dtype

    tensor = torch.zeros(shape, dtype=dtype)
    for i in range(len(lst)):
        tensor[i] = lst[i]
    return tensor

def nig_kl_div(tau_adv, beta_adv, tau_post, beta_post):
    """Compute the KL divergence between 2 NIG distributions"""
    return kl_divergence(tau_adv, tau_post) + kl_divergence(beta_adv, beta_post)