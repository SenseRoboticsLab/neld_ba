"""Register every frame with a learnt trajectory and stack them into one map.

Give it a trajectory the trainer wrote and it finds the rest on its own: the job directory
is the trajectory's grandparent, which holds the config that names the dataset and the
normalisation.json that says what the run's cube units meant in metres.

    python src/make_raw_map.py output/20260629/test_job/traj/iter77000.txt

The trajectory is in cube units, the frames on disk are in metres. The poses are taken back
to metres before registering, so the map that comes out is metric and can be compared
against a ground truth map directly.
"""
import os
os.environ["XDG_SESSION_TYPE"] = "x11"
import toml
import argparse
import numpy as np
import open3d as o3d

from pathlib import Path

from training_helpers.load_helper import load_point_cloud
from training_helpers.normalisation import register_a_frame

VOXEL_SIZE_METER = 0.05     # downsampling size of the combined map, in metres


def find_job_inputs(traj_path: Path):
    """Work out what a trajectory belongs to from its path
    A trajectory lives at <job_dir>/traj/iterNNNN.txt, so the job directory is two levels
    up and holds the config that names the dataset.
    """
    job_dir = traj_path.parents[1]

    config_path = next(job_dir.glob("*.toml"), None)
    if config_path is None:
        raise FileNotFoundError(
            f"no .toml config in {job_dir} -- the trajectory is expected to sit at "
            "<job_dir>/traj/<iteration>.txt, next to the config the run was started with"
        )
    config = toml.load(config_path)
    dataset_dir = Path(config['data_io']['input_dir'])
    print(f"config {config_path} points at dataset {dataset_dir}")

    return dataset_dir


def load_trajectory_as_metric(traj_path: Path):
    """Read a learnt TUM trajectory, which is already in metres.
    """
    poses = {}
    with open(str(traj_path), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or line.startswith("#"):
                continue

            poses[float(parts[0])] = (
                np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]),
            )

    if len(poses) == 0:
        raise ValueError(f"no poses could be read from {traj_path}")

    print(f"loaded {len(poses)} poses from {traj_path}")
    return poses


def register_all_frames(dataset_dir: Path, poses: dict):
    """Register every frame that the trajectory has a pose for, into one metric cloud."""
    point_cloud_dir = dataset_dir / 'pcd'
    point_cloud_paths_list = sorted(point_cloud_dir.glob("*.pcd"))
    if len(point_cloud_paths_list) == 0:
        raise FileNotFoundError(f"no .pcd files found in {point_cloud_dir}")

    points_list = []
    for point_cloud_file in point_cloud_paths_list:
        timestamp = float(point_cloud_file.stem)

        # a trajectory only covers the frames the run trained on, skip anything else
        if timestamp not in poses:
            print(f"no pose in the trajectory for frame {point_cloud_file.name}, skipping")
            continue

        sensor_position, sensor_quaternion = poses[timestamp]
        points_sensor = load_point_cloud(point_cloud_file)   # metres, sensor frame
        points_list.append(register_a_frame(points_sensor[:, 0:3], sensor_position, sensor_quaternion))

    if len(points_list) == 0:
        raise ValueError(
            f"no frame in {point_cloud_dir} could be matched to a pose in the trajectory -- "
            "check that the trajectory belongs to a run over this dataset"
        )

    combined_points = np.vstack(points_list)
    print(f"registered {len(points_list)} frames into {len(combined_points)} points")
    return combined_points


def downsample_and_save(points: np.ndarray, out_path: Path, voxel_size_meter: float):
    """Voxel downsample the map and write it as a binary .pcd."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size_meter)
    print(f"downsampled to {len(point_cloud.points)} points at {voxel_size_meter} m")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), point_cloud, write_ascii=False)
    print(f"map saved to {out_path}")
    return point_cloud


def main(traj_path: Path, voxel_size_meter: float):
    dataset_dir = find_job_inputs(traj_path)
    poses = load_trajectory_as_metric(traj_path)
    combined_points = register_all_frames(dataset_dir, poses)

    # the map belongs to the trajectory that built it, so name it after that iteration
    out_path = traj_path.parents[1] / 'raw_map' / f"{traj_path.stem}.pcd"
    downsample_and_save(combined_points, out_path, voxel_size_meter)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="register every frame with a learnt trajectory into one metric map")
    parser.add_argument("traj_path", help="trajectory .txt written by a training run, at <job_dir>/traj/<iteration>.txt", type=str)
    parser.add_argument("-v", "--voxel_size_meter", help="voxel downsampling size of the map, in metres",
                        type=float, default=VOXEL_SIZE_METER)
    args = parser.parse_args()
    main(Path(args.traj_path), args.voxel_size_meter)
