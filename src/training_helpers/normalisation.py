"""Measure and apply the transform that maps a metric scene into the [-1, 1] cube.

Everything on disk stays in metres -- the frames, the poses, the voxel size in the
config. The trainer measures this transform from the frames it has just loaded and
applies it on the fly, so no dataset on disk carries a normalisation chosen by an
earlier, separate pass that can drift out of step with it.

The scaling is set from the 99.995 / 0.005 percentile span rather than the true extent,
so that a handful of stray returns cannot shrink the whole scene away from the cube the
model samples in. The extreme tail (~0.01% of points) is left outside the cube by design.
A single isotropic factor is used so the geometry is not distorted.

The factors are measured per run, so they are written to the job output directory as
normalisation.json for anything downstream that needs to get back to metres.
"""
import json
import logging
import numpy as np

from pathlib import Path
from einops import rearrange
from scipy.spatial.transform import Rotation

NORMALISATION_FILE_NAME = 'normalisation.json'


def register_a_frame(points_sensor: np.ndarray, sensor_position: np.ndarray, sensor_quaternion: np.ndarray):
    """Take a frame from its sensor frame into the world frame."""
    rotation_matrix = Rotation.from_quat(sensor_quaternion).as_matrix()
    rotation_matrix = rearrange(rotation_matrix, 'a b -> b a')  # take transpose
    points_world = points_sensor @ rotation_matrix + sensor_position
    return points_world


def unregister_a_frame(points_world: np.ndarray, sensor_position: np.ndarray, sensor_quaternion: np.ndarray):
    """Take a frame from the world frame back into its sensor frame."""
    rotation_matrix = Rotation.from_quat(sensor_quaternion).as_matrix()
    points_sensor = (points_world - sensor_position) @ rotation_matrix
    return points_sensor


def compute_normalisation_factors(point_clouds: list, sensor_positions: list, sensor_quaternions: list):
    """Register every frame and measure the transform that maps the scene into the cube.
    Args:
        point_clouds: per-frame [n, 3] (or [n, 4], the scalar column is ignored) arrays
            in the sensor frame, in metres.
        sensor_positions: per-frame [3] positions, in metres.
        sensor_quaternions: per-frame [4] quaternions, [qx, qy, qz, qw].

    Returns:
        (scaling_factor, centre_point), the divisor and the world-frame centre that map
        a metric point into the [-1, 1] cube.
    """
    if len(point_clouds) == 0:
        raise ValueError("no frames to measure the normalisation from")

    point_cloud_world = [
        register_a_frame(points[:, 0:3], position, quaternion)
        for points, position, quaternion in zip(point_clouds, sensor_positions, sensor_quaternions)
    ]
    point_cloud_world = np.vstack(point_cloud_world)

    # this normalisation method has to be updated if we have sensors surrounding the object
    top_points = np.percentile(point_cloud_world, q=99.995, axis=0)
    bottom_points = np.percentile(point_cloud_world, q=0.005, axis=0)
    scaling_factor = float(np.max(top_points - bottom_points) * 0.5)
    centre_point = (top_points + bottom_points) * 0.5
    del point_cloud_world  # the stacked copy is no longer needed

    if scaling_factor == 0.0:
        raise ValueError("degenerate point cloud: every point sits on the centre point")

    return scaling_factor, centre_point


def normalise_points_sensor_frame(points_sensor: np.ndarray, scaling_factor: float):
    """Scale a frame into cube units, in its own sensor frame.

    The centre shift cancels out in the sensor frame, so only the scaling applies here
    and the pose quaternion is left untouched.
    """
    return points_sensor / scaling_factor


def normalise_points_world_frame(points_world: np.ndarray, scaling_factor: float, centre_point: np.ndarray):
    """Move world-frame points into cube units, the inverse of denormalise_points_world_frame.

    Takes any world-frame point, whether that is a registered return or a sensor position.
    """
    return (points_world - centre_point) / scaling_factor


def denormalise_points_world_frame(points_cube: np.ndarray, scaling_factor: float, centre_point: np.ndarray):
    """Take world-frame points in cube units back to metres."""
    return points_cube * scaling_factor + centre_point


def save_normalisation(out_dir: Path, scaling_factor: float, centre_point: np.ndarray, num_frames: int):
    """Record the factors this run measured, so downstream steps can get back to metres."""
    out_path = Path(out_dir) / NORMALISATION_FILE_NAME
    normalisation = {
        'scaling_factor': float(scaling_factor),
        'centroid_translation': np.asarray(centre_point).tolist(),
        'units': 'meters',
        'num_frames': int(num_frames),
    }
    with open(str(out_path), 'w') as f:
        json.dump(normalisation, f, indent=4)
    logging.info(f"normalisation saved to {out_path}")
    return out_path


def load_normalisation(job_dir: Path):
    """Read back the factors a training run measured.

    Returns:
        (scaling_factor, centre_point) as written by save_normalisation.
    """
    normalisation_path = Path(job_dir) / NORMALISATION_FILE_NAME
    if not normalisation_path.is_file():
        raise FileNotFoundError(
            f"no {NORMALISATION_FILE_NAME} in {job_dir} -- it is written by the training run, "
            "so a job trained before the normalisation moved into the trainer will not have one"
        )
    with open(str(normalisation_path), 'r') as f:
        normalisation = json.load(f)
    return float(normalisation['scaling_factor']), np.array(normalisation['centroid_translation'])
