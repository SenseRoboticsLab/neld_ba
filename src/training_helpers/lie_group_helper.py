### The following code are given by nerf--, with slight modification for this project
import numpy as np
import torch
from scipy.spatial.transform import Rotation as RotLib
from einops import rearrange


def SO3_to_quat(R):
    """
    :param R:  (N, 3, 3) or (3, 3) np
    :return:   (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def quat_to_SO3(quat):
    """
    :param quat:    (N, 4, ) or (4, ) np
    :return:        (N, 3, 3) or (3, 3) np
    """
    x = RotLib.from_quat(quat)
    R = x.as_matrix()
    return R


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def skew2vec(skew):
    """
    Convert a skew-symmetric matrix to a vector (so(3)).
    :param skew: (3, 3) skew-symmetric matrix, torch tensor
    :return:     (3, ) vector, torch tensor
    """
    v = torch.tensor([skew[2, 1], skew[0, 2], skew[1, 0]], device=skew.device)
    return v

def Log(R):
    """
    Convert an SO(3) matrix to an so(3) vector (axis-angle).
    :param R: (3, 3) rotation matrix, torch tensor
    :return:  (3, ) axis-angle vector, torch tensor
    """
    eps = 1e-6  # Small value to avoid numerical instability
    trace = R.trace()
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))  # Angle of rotation
    
    if theta < eps:  # Near zero rotation, axis is undefined but can be approximated
        return torch.zeros(3, dtype=torch.float32, device=R.device)
    
    # Skew-symmetric part of log(R)
    skew_r = (R - R.T) / (2 * torch.sin(theta))
    r = theta * skew2vec(skew_r)  # Axis-angle representation
    return r


def make_c2w_skew(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    # c2w = convert3x4_4x4(c2w)  # (4, 4)
    c2w = torch.cat([c2w, torch.tensor([[0,0,0,1]], dtype=c2w.dtype, device=c2w.device)], dim=0)  # (4, 4)
    return c2w

def make_c2w_quaterion(q, t):
    """
    :param r:  (4, ) quaterion              torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    x = RotLib.from_quat(q)
    R = torch.tensor(x.as_matrix())
    c2w = torch.eye(4, dtype = torch.float32)
    # Populate c2w with rotation matrix and translation
    c2w[0:3, 0:3] = R
    c2w[0:3,3] = t
    return c2w

def quaterion2vec(q):
    x = RotLib.from_quat(q)
    R = torch.tensor(x.as_matrix())
    r = Log(R)
    return r

def make_c2w_xy_plane_skew(r, t):
    """
    Construct a 4x4 camera-to-world transformation matrix for a pose in the XY plane.
    
    :param r: (1,) tensor, learnable rotation angle in radians around Z-axis (e.g., nn.Parameter)
    :param t: (2,) tensor, translation vector (x, y) in XY plane
    :return: (4, 4) tensor, camera-to-world transformation matrix
    """
    # Initialize 4x4 identity matrix on the same device as r
    c2w = torch.eye(4, dtype=torch.float32, device=r.device)
    
    # Compute 2D rotation matrix components directly from r (keep it a tensor)
    cos_r = torch.cos(r)  # (1,)
    sin_r = torch.sin(r)  # (1,)
    
    # Construct the 2x2 rotation submatrix with gradient tracking
    c2w[0:2, 0:2] = torch.stack([
        torch.stack([cos_r[0], -sin_r[0]]),  # First row
        torch.stack([sin_r[0],  cos_r[0]])   # Second row
    ], dim=0)  # (2, 2)
    
    # Set the translation in XY plane
    c2w[0:2, 3] = t
    return c2w

if __name__ == "__main__":
    import torch
    import open3d as o3d
    import numpy as np

    # Function definitions (assuming your `vec2skew` and `Exp` are defined as in your code)

    # Create a 3D object to visualize (e.g., a coordinate frame)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # Function to update the rotation matrix with the given parameter t
    def rotation_example(t):
        r = torch.tensor([t, 0, 0], dtype=torch.float32)
        R = Exp(r).numpy()  # Get the rotation matrix from your function
        return R

    # Visualize the rotation in Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(mesh)

    for t in np.linspace(-2* np.pi, 2 * np.pi, 100):  # Vary t from 0 to 2π
        R_t = rotation_example(t)  # Get the rotation matrix at time t
        
        # Apply rotation to the mesh
        mesh.rotate(R_t, center=(0, 0, 0))
        
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

    vis.run()
    vis.destroy_window()