"""Peek at the point clouds the trainer sees, and at what the normalisation does to them.

Frames and poses are loaded in metric world scale, the normalisation into the [-1, 1] cube
is measured here rather than taken from the training loader, and every frame is registered,
scaled and stacked into one cloud drawn inside a red wireframe of that cube. Each frame gets
its own colour, so a misregistered frame stands out.

Running the normalisation in the open is the point: this is the tool for deciding whether
the factors the trainer would measure really do put the scene where the model samples.

    python utility/peek_training_data.py data/quad_hard
"""
import os
os.environ["XDG_SESSION_TYPE"] = "x11"
import argparse
import logging
import sys
import numpy as np
import open3d as o3d

from pathlib import Path
from matplotlib import colormaps

# the training helpers live under src/, so make them importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training_helpers.normalisation import (compute_normalisation_factors, register_a_frame,
                                            normalise_points_world_frame)
from training_helpers.load_helper import (load_point_cloud, load_pose_txt_as_dict,
                                          timestamp_to_digits, voxel_downsample)

CUBE_BOUND = 1.0            # the model samples inside the [-1, 1] cube
VOXEL_SIZE_METER = 0.05     # downsampling size, in metric space


def read_dataset(input_dir: Path):
    """Load every frame of a dataset with its pose, in metric world scale.

    Goes through the low-level loaders rather than load_data, so this script ends up
    holding the metric frames and can run the normalisation itself.

    Returns:
        (point_clouds, sensor_positions, sensor_quaternions), all in metres, with the
        frames still at full resolution and in their own sensor frame.
    """
    point_cloud_dir = input_dir / 'pcd'
    pose_path = input_dir / 'poses.txt'

    poses_dict = load_pose_txt_as_dict(pose_path)
    point_cloud_paths_list = sorted(point_cloud_dir.glob("*.pcd"))
    if len(point_cloud_paths_list) == 0:
        raise FileNotFoundError(f"no .pcd files found in {point_cloud_dir}")

    point_clouds, sensor_positions, sensor_quaternions = [], [], []
    for point_cloud_file in point_cloud_paths_list:
        timestamp_digits = timestamp_to_digits(point_cloud_file.stem)

        # skip frames that have no corresponding pose, exactly as the trainer does
        if timestamp_digits not in poses_dict:
            logging.warning(f"no pose found for frame {point_cloud_file.name}, skipping")
            continue

        pose = poses_dict[timestamp_digits]
        point_clouds.append(load_point_cloud(point_cloud_file))
        sensor_positions.append(np.array([pose['x'], pose['y'], pose['z'],]))
        sensor_quaternions.append(np.array([pose['qx'], pose['qy'], pose['qz'], pose['qw'],]))

    if len(point_clouds) == 0:
        raise ValueError(
            f"no frame in {point_cloud_dir} could be matched to a pose in {pose_path} -- "
            "check that the .pcd file names are the timestamps of the poses"
        )

    logging.info(f"loaded {len(point_clouds)} frames from {point_cloud_dir}, in metres")
    return point_clouds, sensor_positions, sensor_quaternions


def make_cube_skeleton(bound: float = CUBE_BOUND):
    """Build a red wireframe of the [-bound, bound] cube."""
    cube_vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],   # top face
    ], dtype=np.float64) * bound

    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # top square
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]

    cube = o3d.geometry.LineSet()
    cube.points = o3d.utility.Vector3dVector(cube_vertices)
    cube.lines = o3d.utility.Vector2iVector(cube_edges)
    cube.paint_uniform_color([1, 0, 0])
    return cube


def combine_frames(point_clouds: list, sensor_positions: list, sensor_quaternions: list,
                   scaling_factor: float, centre_point: np.ndarray):
    """Downsample, register and scale every frame into the cube, as one coloured cloud.

    The downsampling happens in metres and after the factors were measured, which is the
    order the training loader uses.
    """
    num_frames = len(point_clouds)

    points_list = []
    colours_list = []
    for frame_index, (points_sensor, sensor_position, sensor_quaternion) in enumerate(
            zip(point_clouds, sensor_positions, sensor_quaternions)):
        points_sensor = voxel_downsample(points_sensor, VOXEL_SIZE_METER)

        # register in metres, then take the registered frame into cube units
        points_world = register_a_frame(points_sensor[:, 0:3], sensor_position, sensor_quaternion)
        points_cube = normalise_points_world_frame(points_world, scaling_factor, centre_point)
        points_list.append(points_cube)

        # one hue per frame, walking the colour map from the first frame to the last
        rgb = np.array(colormaps['turbo'](frame_index / max(num_frames - 1, 1))[:3])
        colours_list.append(np.tile(rgb, (points_cube.shape[0], 1)))

    combined_points = np.vstack(points_list)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(combined_points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(np.vstack(colours_list))

    logging.info(f"registered {num_frames} frames, {len(combined_points)} points, in cube units")
    return point_cloud, combined_points


def report_cube_fit(points: np.ndarray, sensor_positions: np.ndarray, bound: float = CUBE_BOUND):
    """Say how much of the scene the cube actually holds -- the reason to look at this plot.

    Both arguments are in cube units.
    """
    outside = np.any(np.abs(points) > bound, axis=1)
    logging.info(f"point extent: min {points.min(axis=0)}, max {points.max(axis=0)}")
    logging.info(f"{outside.sum()}/{len(points)} points ({100 * outside.mean():.3f}%) fall outside the cube")

    sensors_outside = np.any(np.abs(sensor_positions) > bound, axis=1).sum()
    if sensors_outside > 0:
        logging.warning(f"{sensors_outside}/{len(sensor_positions)} sensor positions sit outside the cube")


def main(input_dir: Path):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # load in world scale, then measure the normalisation off the full clouds, as the loader does
    point_clouds, sensor_positions, sensor_quaternions = read_dataset(input_dir)
    scaling_factor, centre_point = compute_normalisation_factors(
        point_clouds, sensor_positions, sensor_quaternions)
    logging.info(f"measured scaling factor {scaling_factor} m about centre {centre_point}")

    # assemble and inspect everything in cube space
    point_cloud, combined_points = combine_frames(
        point_clouds, sensor_positions, sensor_quaternions, scaling_factor, centre_point)
    sensor_positions_cube = normalise_points_world_frame(
        np.vstack(sensor_positions), scaling_factor, centre_point)
    report_cube_fit(combined_points, sensor_positions_cube)

    o3d.visualization.draw_geometries([point_cloud, make_cube_skeleton()], window_name=input_dir.name)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualise the registered training point clouds inside the [-1,1] cube")
    parser.add_argument("input_dir", help="dataset directory, holding pcd/ and poses.txt, in metres", type=str)
    args = parser.parse_args()
    main(Path(args.input_dir))
