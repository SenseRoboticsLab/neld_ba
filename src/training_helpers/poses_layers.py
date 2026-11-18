### The following code are mainly from nerf-- [https://nerfmm.active.vision/], with modification for this project
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from training_helpers.lie_group_helper import make_c2w_skew, make_c2w_xy_plane_skew
from training_helpers.normalisation import denormalise_points_world_frame

def vec2quaternion(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert a skew (axis–angle) vector to a quaternion.
    
    Args:
        r: Tensor of shape (..., 3), the axis–angle vectors.
           The rotation angle is theta = ||r||, axis = r/theta.
        eps: Small threshold to avoid division by zero.
    
    Returns:
        q: Tensor of shape (..., 4), quaternions in (x, y, z, w) format.
    """
    # Compute norm (rotation angle)
    theta = torch.norm(r, dim=-1, keepdim=True)  # (..., 1)
    half_theta = theta * 0.5                       # (..., 1)

    # Compute scalar part
    w = torch.cos(half_theta)                      # (..., 1)

    # Compute axis-scaled vector part robustly
    # scale = sin(theta/2)/theta, but handle theta ≈ 0
    scale = torch.where(theta > eps, torch.sin(half_theta) / theta, 0.5 + 0.0 * theta)  # lim θ→0 sin(θ/2)/θ = 1/2
    xyz = r * scale                                # (..., 3)
    return torch.cat((xyz, w), dim=-1)

class PoseLayer(nn.Module):
    def __init__(self, num_cams, scaling_factor=None, centre_point=None, timestamps=None,
                 init_r=None, init_t=None, device='cpu'):
        """
        :param num_cams:
        :param scaling_factor: metres per cube unit, as measured by the run's normalisation
        :param centre_point: the scene centre in metres, as measured by the run's normalisation
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super().__init__()
        self.num_cams = num_cams
        self.device = device
        self.timestamps = timestamps
        self.scaling_factor = scaling_factor
        self.centre_point = None if centre_point is None else np.asarray(centre_point)
        r_values = init_r if init_r is not None else torch.zeros(size=(num_cams, 3), dtype=torch.float32)
        t_values = init_t if init_t is not None else torch.zeros(size=(num_cams, 3), dtype=torch.float32)
        self.r = nn.Parameter(r_values, requires_grad=False)
        self.t = nn.Parameter(t_values, requires_grad=False)  # (N,2)

    def forward(self, sensor_id):
        # return identity matrix if sensor idis pseudo sensor
        if sensor_id >= self.num_cams:
            return torch.eye(4, device=self.device)
        r = self.r[sensor_id]  # (3, ) axis-angle
        t = self.t[sensor_id]  # (3, )
        c2w = make_c2w_skew(r, t)  # (4, 4)
        return c2w
    
    def set_trainable(self, trainable_r: bool, trainable_t: bool):
        self.r.requires_grad = trainable_r
        self.t.requires_grad = trainable_t

    def check_trainable(self):
        return self.r.requires_grad, self.t.requires_grad

    def save_as_tum(self, out_path):
        """Write the learnt trajectory as TUM text, in metres.
        """
        if self.scaling_factor is None or self.centre_point is None:
            raise ValueError(
                "this PoseLayer was built without a normalisation, so its trajectory cannot "
                "be put back into metres -- pass scaling_factor and centre_point to build it"
            )

        traj_arr = np.zeros((self.num_cams, 8))
        for sensor_id in range(self.num_cams):
            sensor_quaternion = vec2quaternion(self.r[sensor_id,:]).detach().cpu().numpy()
            position_cube = self.t[sensor_id,:].detach().cpu().numpy()
            timestamp = self.timestamps[sensor_id]
            traj_arr[sensor_id,0] = timestamp
            traj_arr[sensor_id,1:4] = denormalise_points_world_frame(
                position_cube, self.scaling_factor, self.centre_point)
            traj_arr[sensor_id,4:8] = sensor_quaternion
        fmt = '%.18e'   # one format specifier for each column
        np.savetxt(out_path, traj_arr, fmt=[fmt]*8, delimiter=' ')
        return