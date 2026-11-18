import torch
from torch import nn
from einops import rearrange

def render_rays(alpha_pred, magnitudes, num_rays, num_bins=128, device='cpu') -> torch.Tensor:
    alpha_pred = rearrange(alpha_pred, '(n b) -> n b', n=num_rays, b=num_bins)

    # compute predicted cumulative transmittance
    K = torch.cumprod((1 - alpha_pred), 1)   # K is just a temporary variable
    T_pred = torch.ones(num_rays, num_bins, device=device)
    T_pred[:, 1:] = K[:, :-1]
    T_pred.clamp_(min=1e-7, max=1-1e-7) # clamping variables like this might make it difficult to debug logic bugs, idk

    # compute predicted termination distribution
    h_pred = T_pred*alpha_pred
    h_pred = h_pred / h_pred.sum(dim=1, keepdim=True)
    h_pred.clamp_(min=1e-7)
    
    # compute expected depth
    d_pred = torch.sum(h_pred*magnitudes, axis = 1)
    return h_pred, d_pred


def get_target_term_dist(magnitude, ray_distance, num_rays, device='cpu') -> torch.Tensor:
    # minus magnitude by ray distance
    termination_centered_magnitude = magnitude - ray_distance
    mask = termination_centered_magnitude.clone()
    mask[termination_centered_magnitude < 0] = float('inf')
    hot_indices = mask.argmin(dim=1)  # choosing smallest positive element in each row to be hot

    # create one-hot rows as target termination distribution
    h_target = torch.zeros_like(termination_centered_magnitude, device=device)
    h_target[torch.arange(num_rays), hot_indices] = 1
    h_target.clamp_(min=1e-7, max=1-1e-7)
    return h_target