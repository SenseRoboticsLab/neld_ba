import torch
from einops import rearrange

def find_aabb_intersections(positions, directions, near_plane: float, far_plane: float,
                            padding: float = 0.004, eps: float = 1e-8):
    """
    Exact intersection of each ray with the axis aligned box [-1, 1]^3, grown by `padding`.

    Each pair of opposite box faces is a "slab", and a ray is inside the box only where all
    three slab intervals overlap.  Taking the last entry and the first exit over the three
    axes therefore gives the entry and exit distance in closed form, with no marching.

    Parameters:
    positions:  [num_rays, 3] ray origins, in cube units
    directions: [num_rays, 3] ray directions, assumed unit length, so that a distance along
                a ray is the same number as the multiplier on its direction
    near_plane: closest distance to sample at, also the fallback near for rays that miss
    far_plane:  furthest distance to sample at, also the fallback far for rays that miss

    Returns:
    near: [num_rays] distance to start sampling at
    far:  [num_rays] distance to stop sampling at
    hit:  [num_rays] bool, False on rays that never enter the box
    """
    boundary = 1.0 + padding

    # a zero direction component makes the ray parallel to that slab and its two crossing
    # distances infinite.  Nudging the component off zero keeps every quotient finite, which
    # rules out the 0 * inf = NaN that appears when an origin lies on a slab it is parallel to
    signs = torch.where(directions >= 0, 1.0, -1.0)
    safe_directions = torch.where(directions.abs() < eps, eps * signs, directions)

    # distance to both faces of every slab
    t_lower = (-boundary - positions) / safe_directions   # [num_rays, 3]
    t_upper = (boundary - positions) / safe_directions    # [num_rays, 3]

    # the ray is inside the box from its last slab entry until its first slab exit
    t_enter = torch.minimum(t_lower, t_upper).amax(dim=-1)   # [num_rays]
    t_exit = torch.maximum(t_lower, t_upper).amin(dim=-1)    # [num_rays]

    # a ray misses when that interval is empty, or when the whole box is behind the sensor
    hit = t_exit >= torch.clamp(t_enter, min=0.0)

    # the interval only decides where the samples go, it is not part of the rendering, so it
    # is detached: pose gradients reach the loss through the sampling positions instead
    t_enter, t_exit = t_enter.detach(), t_exit.detach()

    # sensors normally sit inside the box, which puts their entry distance behind them
    near = torch.clamp(t_enter, min=near_plane)
    far = torch.clamp(t_exit, max=far_plane)
    far = torch.maximum(far, near + eps)    # keep every bin width strictly positive

    # rays that miss the box fall back to the configured planes
    near = torch.where(hit, near, torch.full_like(near, near_plane))
    far = torch.where(hit, far, torch.full_like(far, far_plane))
    return near, far, hit

def get_spacing(num_bins, num_rays):
    """return a [num_points, num_bins] pytorch tensor
    Give perturbated even spacing value from 0 to 1 along num_bin dimension
    """
    # generate even spacing
    t = torch.stack([torch.linspace(0.00001, 0.99999, num_bins + 1) for _ in range(num_rays)], dim=0)     # [num_rays, num_bins]
    t = t[:,:-1]    # discard the last column

    # preterb the spacing
    u = torch.rand(t.shape)
    t = t + (0.99998 / num_bins) * u  # [num_rays, num_bins], using x = x + y to avoid gradient problem, constant is binded with above linspace()
    return t

def get_spacing_scaled(near_point_dists, far_point_dists, num_bins):
    """
    near_point_dists: [N] tensor
    far_point_dists:  [N] tensor
    Returns: [N, num_bins] tensor
    """
    device = near_point_dists.device

    # Create a [num_bins] linspace template, the [:-1] at the end select the first num_bins element, so that difference between values are 1 / num_bins
    t_vals = torch.linspace(0, 1, steps=num_bins + 1, device=device)[:-1]  # [num_bins]
    
    # Reshape for broadcasting
    t_vals = t_vals[None, :]                   # [1, num_bins]
    near = near_point_dists[:, None]           # [N, 1]
    far = far_point_dists[:, None]             # [N, 1]

    # Interpolate linearly
    t = near * (1 - t_vals) + far * t_vals     # [N, num_bins]

    # Perturb spacing (like in NeRF)
    u = torch.rand_like(t)
    delta = (far - near) / num_bins            # [N, 1]
    magnitudes = t + delta * u                 # [N, num_bins]
    return magnitudes

def get_sampling_pos_dir(sensor_poses, ray_direction, near_plane: float, far_plane: float, num_bins: int, num_rays: int, device='cpu') -> torch.Tensor:
    # extract sensor pose 
    sensor_positions = sensor_poses[:,0:3,3]
    sensor_rotations = sensor_poses[:,0:3, 0:3]
    sensor_rotations = rearrange(sensor_rotations, 'n a b -> n b a')

    # rotate ray_direction
    ray_direction_rotated = torch.matmul(ray_direction.unsqueeze(1), sensor_rotations).squeeze(1)

    # the hit mask is dropped here: missed rays already fall back to [near_plane, far_plane],
    # which is what they used to get, and it is available if the loss should mask them out
    near_point_dists, far_point_dists, _ = find_aabb_intersections(positions=sensor_positions,
                                                            directions=ray_direction_rotated,
                                                            near_plane=near_plane,
                                                            far_plane=far_plane)

    # apply truncated inverse sigmoid to concentrate samplings around termination
    magnitudes = get_spacing_scaled(near_point_dists, far_point_dists, num_bins)

    # calculate separation between neighboring positions, assign a large deparation to last sample along all rays
    thousands = torch.full((num_rays, 1), 1000, device=device)
    delta = magnitudes[:, 1:] - magnitudes[:, :-1]
    delta = torch.cat([delta, thousands], dim=1)  # [num_rays, num_bins]
    delta = delta.flatten() # same as using: rearrange(delta, 'n b -> (n b)')

    # reshape for computing sampling positions
    magnitudes_reshaped = rearrange(magnitudes, 'n b -> (n b) 1')

    # [num_rays, 3] -> [(num_rays * num_bins), 3]
    ray_direction_rotated = ray_direction_rotated[:, None, :].expand(-1, num_bins, -1)
    ray_direction_rotated = ray_direction_rotated.reshape(-1, 3)

    # [num_rays, 3] -> [(num_rays * num_bins), 3]
    sensor_positions = sensor_positions[:, None, :].expand(-1, num_bins, -1)
    sensor_positions = sensor_positions.reshape(-1, 3)

    # Calculate sampling position
    sampling_positions = magnitudes_reshaped*ray_direction_rotated + sensor_positions # [(num_rays*num_bins), 3]
    return sampling_positions, ray_direction_rotated, delta, magnitudes

def term_dist_to_hist(term_distribution: torch.Tensor, num_fine_samples: int) -> torch.Tensor:
    '''
    Convert batched probability distributions to integer histograms that sum to total_count per row.
    Args:
        probs: Tensor of shape [B, N], where each row sums to 1
        total_count: int, target histogram count per row
    Returns:
        hist: Tensor of shape [B, N], dtype=int32, each row sums to total_count
    '''
    # term_distribution = term_distribution[:,1:]
    # term_distribution = term_distribution / term_distribution.sum(dim=1, keepdim=True)

    num_rays, _ = term_distribution.shape

    # Multiply termination distribution by number of fine samples to get the rough number of fine samples needed
    raw_fine_samples_hist = term_distribution * num_fine_samples
    base_fine_samples_hist = torch.floor(raw_fine_samples_hist).to(torch.int32)       # [B, N]
    remainder = raw_fine_samples_hist - base_fine_samples_hist # [B, N]

    # The above computation floors the number of fine samples, so we need to allocate deficit samples to top k bins in base_fine_sample_hist
    deficit = num_fine_samples - base_fine_samples_hist.sum(dim=1)        # [B]

    # For each row, get top-k indices from remainder and add 1 (note that deficit is certainly not more than num_bins)
    fine_samples_hist = base_fine_samples_hist.clone()
    for n in range(num_rays):
        if deficit[n] > 0:
            _, top_k_idx = torch.topk(remainder[n], k=deficit[n].item())
            fine_samples_hist[n, top_k_idx] += 1
    return fine_samples_hist

# the following function is modified from nerf
def sample_pdf(bin_edges, weights, num_bins_fine):
    """
    bins:     [64, 129] tensor (bin edges)
    weights:  [64, 128] tensor (probability weights)
    N_samples: number of samples to draw per row
    det:      whether to sample deterministically (linspace) or randomly
    """
    # Add small epsilon to prevent NaNs
    weights = weights + 1e-7
    prob_mass_dist = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Compute CDF
    cdf = torch.cumsum(prob_mass_dist, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # prepend 0 at start

    # Sample u ~ Uniform[0,1]
    uniform_spacing = torch.linspace(0., 1., steps=num_bins_fine, device=weights.device)
    uniform_spacing = uniform_spacing.expand(cdf.shape[0], num_bins_fine)

    # Invert CDF
    bin_edge_idxs = torch.searchsorted(cdf, uniform_spacing, right=True) # searchsorted() finds the first index in cdf where cdf >= u
    below = torch.clamp(bin_edge_idxs - 1, min=0)
    above = torch.clamp(bin_edge_idxs, max=cdf.shape[-1] - 1)
    bin_edge_indx_gather = torch.stack([below, above], dim=-1)  # shape [64, N_samples, 2]

    # gather means that for each fine bin, you get their bin edges and their CDF balue at bin edges
    cdf_gathered = torch.gather(cdf.unsqueeze(1).expand(-1, num_bins_fine, -1), 2, bin_edge_indx_gather)
    bin_edges_gathered = torch.gather(bin_edges.unsqueeze(1).expand(-1, num_bins_fine, -1), 2, bin_edge_indx_gather)

    # denom can be think of the "distance" in the [0,1] interval between cdf
    denom = cdf_gathered[..., 1] - cdf_gathered[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    # Compute samples
    t = (uniform_spacing - cdf_gathered[..., 0]) / denom
    samples = bin_edges_gathered[..., 0] + t * (bin_edges_gathered[..., 1] - bin_edges_gathered[..., 0])
    return samples  # shape [64, N_samples]

def get_fine_sampling_pos_dir(magnitudes_fine, sensor_poses, ray_directions, device):
    num_rays, num_bins = magnitudes_fine.shape

    # extract sensor pose 
    sensor_positions = sensor_poses[:,0:3,3]
    sensor_rotations = sensor_poses[:,0:3, 0:3]
    sensor_rotations = rearrange(sensor_rotations, 'n a b -> n b a')

    # calculate separation between neighboring positions, assign a large deparation to last sample along all rays
    thousands = torch.full((num_rays, 1), 1000, device=device)
    delta_fine = magnitudes_fine[:, 1:] - magnitudes_fine[:, :-1]
    delta_fine = torch.cat([delta_fine, thousands], dim=1)  # [num_rays, num_bins]
    delta_fine = delta_fine.flatten() # same as using: rearrange(delta, 'n b -> (n b)')
    magnitudes_reshaped = rearrange(magnitudes_fine, 'n b -> (n b) 1')

    # rotate ray direction and reshape 
    ray_direction_rotated = torch.matmul(ray_directions.unsqueeze(1), sensor_rotations).squeeze(1)
    # [num_rays, 3] -> [(num_rays * num_bins), 3]
    ray_direction_rotated = ray_direction_rotated[:, None, :].expand(-1, num_bins, -1) 
    ray_direction_rotated = ray_direction_rotated.reshape(-1, 3)

    # [num_rays, 3] -> [(num_rays * num_bins), 3]
    sensor_positions = sensor_positions[:, None, :].expand(-1, num_bins, -1)
    sensor_positions = sensor_positions.reshape(-1, 3)

    # Calculate sampling position
    sampling_positions_fine = magnitudes_reshaped*ray_direction_rotated + sensor_positions 
    return sampling_positions_fine, ray_direction_rotated, delta_fine