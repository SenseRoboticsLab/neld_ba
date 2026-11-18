import logging
import re
import torch
import numpy as np
import open3d as o3d

from pathlib import Path
from collections import defaultdict
from einops import rearrange
from torch.utils.data import DataLoader
from training_helpers.dataset import RayShuffleSamplerDataset
from training_helpers.lie_group_helper import quaterion2vec
from training_helpers.normalisation import (compute_normalisation_factors, normalise_points_sensor_frame,
                                            normalise_points_world_frame)


def voxel_downsample(point_cloud, voxel_size):
    """Downsample a point cloud by randomly sampling one point per voxel in NumPy.
    """
    # Extract positions
    xyz = point_cloud[:, :3]  # Shape [n, 3]

    # Compute voxel grid indices
    voxel_indices = np.floor(xyz / voxel_size).astype(int)  # Shape [n, 3]

    # Create a unique identifier for each voxel (hash the 3D indices into 1D)
    # Using a large prime number to minimize collisions
    voxel_ids = voxel_indices @ np.array([73856093, 19349663, 83492791])  # Shape [n,]

    # Find unique voxels and their inverse indices
    _, inverse_indices, counts = np.unique(
        voxel_ids, return_inverse=True, return_counts=True
    )

    # Get random indices for each unique voxel
    # Sort points by voxel ID and use cumulative counts to select one point per voxel
    sort_indices = np.argsort(inverse_indices)
    voxel_starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    random_offsets = np.random.randint(0, counts)  # Random offset within each voxel
    random_indices = sort_indices[voxel_starts + random_offsets]

    # Extract the downsampled points using the random indices
    downsampled_points = point_cloud[random_indices]
    return downsampled_points


def get_init_rotation_translation(sensor_quaterion_list, sensor_position_list, num_real_sensors):
    init_r = torch.zeros(size=(num_real_sensors, 3), dtype=torch.float32)
    init_t = torch.zeros(size=(num_real_sensors, 3), dtype=torch.float32)
    for i in range(num_real_sensors):
        quaterion = torch.tensor(sensor_quaterion_list[i])
        v = quaterion2vec(quaterion)
        t = torch.tensor(sensor_position_list[i])
        init_r[i,:] = v
        init_t[i,:] = t
    return init_r, init_t


def timestamp_to_digits(time_stamp: str):
    """Reduce a timestamp to a plain digit string, so every source keys the same way.
    """
    if "e" in time_stamp or "E" in time_stamp:
        time_stamp = f"{float(time_stamp):.9f}"  # back to fixed point before stripping
    return re.sub(r"\D", "", time_stamp)


def load_pose_txt_as_dict(pose_path: Path):
    """Read a TUM-format trajectory: timestamp x y z qx qy qz qw (one pose per line).
    
    Returns a dict keyed by the timestamp digits, matching load_point_cloud()'s keying.
    """
    poses_dict = {}
    with open(str(pose_path), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or line.startswith("#"):
                continue

            poses_dict[timestamp_to_digits(parts[0])] = {
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "qx": float(parts[4]),
                "qy": float(parts[5]),
                "qz": float(parts[6]),
                "qw": float(parts[7]),
            }

    if len(poses_dict) == 0:
        raise ValueError(f"no poses could be read from {pose_path}")
    return poses_dict


def load_point_cloud(pcd_path: Path):
    """Load a .pcd as an [n, 4] array of [x, y, z], in the sensor frame and in metres.
    """
    cloud = o3d.t.io.read_point_cloud(str(pcd_path))
    xyz = cloud.point.positions.numpy().astype(np.float64)
    if xyz.shape[0] == 0:
        raise ValueError(f"no points read from {pcd_path}")
    return xyz


def load_data(input_dir: str, vox_size_meter: float):
    """Load a metric dataset and scale it into the [-1, 1] cube the model samples in.

    Expected layout:
        <input_dir>/pcd/<timestamp>.pcd     point clouds in their sensor frame, metres
        <input_dir>/poses.txt               poses, TUM format, metres

    The normalisation is measured here from the frames that were just loaded.

    Frames and poses are paired on their timestamp digits; a frame with no matching
    pose is skipped, so num_real_sensors counts the frames that were actually kept.

    Returns:
        (data_dict, num_real_sensors, max_distance, scaling_factor, centre_point). The
        rays and sensor positions in data_dict are in [-1, 1]^3 cube units; scaling_factor and
        centre_point are what takes them back to metres.
    """
    data_root = Path(input_dir)
    point_cloud_dir = data_root / 'pcd'
    pose_path = data_root / 'poses.txt'

    # load poses, keyed by timestamp digits
    poses_dict = load_pose_txt_as_dict(pose_path)

    # get point cloud paths
    point_cloud_paths_list = sorted(list(point_cloud_dir.glob("*.pcd")))
    if len(point_cloud_paths_list) == 0:
        raise FileNotFoundError(f"no .pcd files found in {point_cloud_dir}")

    ### load every frame with its pose, all in metres
    timestamps, point_clouds, sensor_positions, sensor_quaternions = [], [], [], []
    for point_cloud_file in point_cloud_paths_list:
        timestamp_digits = timestamp_to_digits(point_cloud_file.stem)

        # skip frames that have no corresponding pose
        if timestamp_digits not in poses_dict:
            logging.warning(f"no pose found for frame {point_cloud_file.name}, skipping")
            continue

        pose = poses_dict[timestamp_digits]
        timestamps.append(int(timestamp_digits))
        point_clouds.append(load_point_cloud(point_cloud_file))
        sensor_positions.append(np.array([pose['x'], pose['y'], pose['z'],]))
        sensor_quaternions.append(np.array([pose['qx'], pose['qy'], pose['qz'], pose['qw'],]))

    num_sensors = len(timestamps)
    if num_sensors == 0:
        raise ValueError(
            f"no frame in {point_cloud_dir} could be matched to a pose in {pose_path} -- "
            "check that the .pcd file names are the timestamps of the poses"
        )
    logging.info(f"loaded {num_sensors} frames from {point_cloud_dir}")

    ### measure the transform into the cube, from the full clouds
    # measured before downsampling on purpose: voxel_downsample keeps a random point per
    # voxel, so measuring after it would shift the factor a little from run to run
    scaling_factor, centre_point = compute_normalisation_factors(point_clouds, sensor_positions, sensor_quaternions)
    logging.info(f"measured scaling factor {scaling_factor} about centre {centre_point}")

    ### downsample, scale into the cube and turn the frames into rays
    data_dict = defaultdict(list)   # for storing and return data
    max_distance = 0    # longest ray, in cube units

    for timestamp, point_cloud, sensor_position, sensor_quaternion in zip(
            timestamps, point_clouds, sensor_positions, sensor_quaternions):
        # voxel downsample in metres
        point_cloud = voxel_downsample(point_cloud, vox_size_meter)

        # drop points sitting on the sensor origin
        point_cloud = point_cloud[np.any(point_cloud[:, 0:3] != 0, axis=1)]

        # scale into cube units, the quaternion is unaffected by either the shift or the scaling
        points_cube = normalise_points_sensor_frame(point_cloud[:, 0:3], scaling_factor)
        sensor_position_cube = normalise_points_world_frame(sensor_position, scaling_factor, centre_point)

        # convert sensor quaternion into vec
        sensor_rotation_vec = np.array(quaterion2vec(sensor_quaternion))

        # convert point cloud to rays for convenience
        ray_distances = rearrange((points_cube[:,0]**2 + points_cube[:,1]**2 + points_cube[:,2]**2)**0.5, 'n -> n 1')
        ray_direction = points_cube / ray_distances
        max_distance = np.maximum(max_distance, np.max(ray_distances))

        # Add ray information and pose to defaultdict
        data_dict["ray_distance"].append(ray_distances)
        data_dict["ray_direction"].append(ray_direction)
        data_dict["sensor_rotation"].append(sensor_rotation_vec)
        data_dict["sensor_position"].append(sensor_position_cube)
        data_dict["timestamps"].append(timestamp)

    return data_dict, num_sensors, max_distance, scaling_factor, centre_point

def load_and_create_dataloader(prep_config, io_config):
    # Load data
    data_dict, num_sensors, max_distance, scaling_factor, centre_point = load_data(
        input_dir=io_config.input_dir,
        vox_size_meter=prep_config.vox_down_meter)

    # init_r, init_t = get_init_rotation_translation(data_dict["sensor_quaternion"], data_dict["sensor_position"], num_sensors)

    # Create sampler
    dataset = RayShuffleSamplerDataset(
        ray_distances=data_dict["ray_distance"], 
        ray_directions=data_dict["ray_direction"], 
    )
    dataloader = DataLoader(dataset, batch_size=prep_config.batch_size, shuffle=True, drop_last=True)
    return dataloader, num_sensors, max_distance, data_dict, scaling_factor, centre_point