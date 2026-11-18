"""Render a map from a trained density field, one view per training frame.

Accepts a nerf model's weights such as:

    python src/make_rendered_map.py output/20260629/test_job/nerf/iter76000.pth

Every LiDAR ray of every frame is re-rendered by the model and its termination distance read off
the density field, giving one mapped point per ray.
"""
import os

from sympy import Ne
os.environ["XDG_SESSION_TYPE"] = "x11"
import re
import toml
import argparse
import numpy as np
import torch
import open3d as o3d

from pathlib import Path
from einops import rearrange, repeat

from training_helpers.load_helper import load_data
from training_helpers.nerf_models import lidarNerfDensity
from training_helpers.poses_layers import PoseLayer
from training_helpers.volume_sampling import find_aabb_intersections, sample_pdf
from training_helpers.embeddings import SurrogateFourierFeature, BARFMasks
from training_helpers.normalisation import denormalise_points_world_frame, load_normalisation

VOXEL_SIZE_METER = 0.05     # downsampling size of the finished map, in metres
NUM_BINS = 256              # coarse samples per ray, the fine pass adds as many again
VOX_DOWN_METER = 0.05       # downsampling of the input frames, one rendered point per surviving ray
RAY_BATCH_SIZE = 256        # rays queried at once, a memory knob that does not change the result


def model_iteration_of(nerf_model_path: Path):
    """Read the training iteration out of a checkpoint name such as iter76000.pth."""
    match = re.search(r'iter(\d+)', nerf_model_path.name)
    if match is None:
        raise ValueError(
            f"cannot tell which iteration {nerf_model_path.name} is from -- checkpoints are "
            "expected to be named iter<iteration>.pth, as the trainer writes them"
        )
    return int(match.group(1))


def find_job_inputs(nerf_model_path: Path):
    """Work out what a checkpoint belongs to, from its path alone.

    A checkpoint lives at <job_dir>/nerf/iterNNNN.pth, so the job directory is two levels up
    and holds the config, the pose checkpoints and the run's normalisation.

    Returns:
        (job_dir, config, dataset_dir, scaling_factor, centre_point).
    """
    job_dir = nerf_model_path.parents[1]

    config_path = next(job_dir.glob("*.toml"), None)
    if config_path is None:
        raise FileNotFoundError(
            f"no .toml config in {job_dir} -- the checkpoint is expected to sit at "
            "<job_dir>/nerf/<iteration>.pth, next to the config the run was started with"
        )
    config = toml.load(config_path)
    dataset_dir = Path(config['data_io']['input_dir'])
    print(f"config {config_path} points at dataset {dataset_dir}")

    scaling_factor, centre_point = load_normalisation(job_dir)
    print(f"run normalised with scaling factor {scaling_factor} m about centre {centre_point}")
    return job_dir, config, dataset_dir, scaling_factor, centre_point


def find_pose_checkpoint(job_dir: Path, model_iteration: int):
    """Find the pose checkpoint closest in iteration to a NeRF checkpoint.

    The two are saved on different cadences, so an exact match is not guaranteed and the
    nearest pose is the best stand-in for the one the model was trained alongside.
    """
    pose_dir = job_dir / 'pose'
    # sorted numerically so a tie picks the earlier iteration rather than whatever glob returned
    pose_paths = sorted(pose_dir.glob("*.pth"), key=model_iteration_of)
    if len(pose_paths) == 0:
        raise FileNotFoundError(f"no pose checkpoints in {pose_dir}")

    closest_path = min(pose_paths, key=lambda path: abs(model_iteration_of(path) - model_iteration))
    iteration_gap = abs(model_iteration_of(closest_path) - model_iteration)
    if iteration_gap == 0:
        print(f"using pose checkpoint {closest_path}")
    else:
        print(f"using pose checkpoint {closest_path}, the closest to the model's iteration {model_iteration}")
    return closest_path


def load_model_eval(nerf_model_path: Path, pose_layer_path: Path, prep_config: dict, num_sensors: int, device):
    """Load the density field, the learnt poses and the embeddings that go with them."""

    nerf_model_eval = lidarNerfDensity(
        embedding_dim_pos=prep_config['pos_emb_dim'],
        embedding_dim_dir=prep_config['dir_emb_dim'],
        hidden_dim=prep_config['hidden_dim'],
        device=device,
    )
    nerf_model_eval.load_state_dict(torch.load(nerf_model_path, weights_only=True))
    nerf_model_eval.to(device=device)
    nerf_model_eval.eval()

    # the learnt poses are in cube units, which is where the render happens
    pose_layer_eval = PoseLayer(num_cams=num_sensors, device=device)
    pose_layer_eval.load_state_dict(torch.load(pose_layer_path, weights_only=True))
    pose_layer_eval.to(device=device)
    pose_layer_eval.eval()

    barf_mask = BARFMasks(prep_config['start_barf_iter'], prep_config['end_barf_iter'],
                          prep_config['pos_emb_dim'], device)
    return nerf_model_eval, pose_layer_eval, SurrogateFourierFeature, barf_mask


def get_sampling_pos_dir_eval(ray_direction_eval, sensor_position, sensor_rotation, num_bins,
                              near_plane: float, far_plane: float, device='cpu'):
    """Place num_bins coarse samples along each ray, spanning where it crosses the cube."""
    num_rays = ray_direction_eval.shape[0]  # IMPORTANT

    # transform ray direction
    ray_direction_eval_rotated = torch.matmul(ray_direction_eval, sensor_rotation)
    sensor_position = sensor_position.unsqueeze(0).expand(num_rays, 3)
    near_point_dists, far_point_dists, _ = find_aabb_intersections(positions=sensor_position,
                                                                  directions=ray_direction_eval_rotated,
                                                                  near_plane=near_plane,
                                                                  far_plane=far_plane)

    # magnitudes is the distance of sensor to each sampling positions, evenly spaced between
    # the two crossings, one row per ray
    t_vals = torch.linspace(0, 1, num_bins, device=device)[None, :]      # [1, num_bins]
    near, far = near_point_dists[:, None], far_point_dists[:, None]      # [num_rays, 1]
    magnitudes = near + (far - near)*t_vals    # [num_rays, num_bins]

    # get delta, which is the distance between neighboring samples, and stack distance of 1000 to last sample
    delta = magnitudes[:, 1:] - magnitudes[:, :-1]
    thousands = torch.full((num_rays, 1), 1000, device=device)
    delta = torch.hstack((delta, thousands))

    # reshape for calculation
    magnitudes_tiled = rearrange(magnitudes, 'n b -> (n b) 1')
    ray_direction_tiled = repeat(ray_direction_eval_rotated, 'n c -> (n b) c', b=num_bins)
    sensor_position_tiled = repeat(sensor_position, 'n c -> (n b) c', b=num_bins)

    # Calculate sampling position
    sampling_positions = magnitudes_tiled*ray_direction_tiled + sensor_position_tiled
    return sampling_positions, ray_direction_tiled, delta, magnitudes


def get_fine_sampling_pos_dir_eval(ray_direction_eval, sensor_rotation, sensor_position, magnitudes_fine, device='cpu'):
    """Place the fine samples, at the distances the coarse pass said were worth looking at."""
    num_rays = ray_direction_eval.shape[0]  # IMPORTANT
    num_bins = magnitudes_fine.shape[1]

    # transform ray direction
    ray_direction_eval_rotated = torch.matmul(ray_direction_eval, sensor_rotation)
    sensor_position = sensor_position.unsqueeze(0).expand(num_rays, 3)

    # get delta, which is the distance between neighboring samples, and stack distance of 1000 to last sample
    thousands = torch.full((num_rays, 1), 1000, device=device)
    deltas_fine = magnitudes_fine[:, 1:] - magnitudes_fine[:, :-1]
    deltas_fine = torch.hstack((deltas_fine, thousands))

    # reshape for calculation
    magnitudes_fine_tiled = rearrange(magnitudes_fine, 'n b -> (n b) 1')
    ray_direction_tiled = repeat(ray_direction_eval_rotated, 'n c -> (n b) c', b=num_bins)
    sensor_position_tiled = repeat(sensor_position, 'n c -> (n b) c', b=num_bins)

    # Calculate sampling position
    sampling_positions = magnitudes_fine_tiled*ray_direction_tiled + sensor_position_tiled
    return sampling_positions, ray_direction_tiled, deltas_fine


def render_a_view(nerf_model_eval, pose_layer_eval, data_dict, SurrogateFourierFeature, barf_mask,
                  prep_config: dict, model_iteration: int, num_bins: int, frame_index: int,
                  near_plane: float, far_plane: float, device='cpu'):
    """Render one frame's rays through the density field.

    Returns:
        [n, 3] array of mapped points, in cube units and in the world frame.
    """
    num_bins_coarse = num_bins
    num_bins_fine = num_bins

    # turn ray direction into batches, and prepare sensor pose
    ray_directions_eval = torch.tensor(data_dict['ray_direction'][frame_index], device=device, dtype=torch.float32)
    ray_directions_eval_batchs = torch.split(ray_directions_eval, RAY_BATCH_SIZE)
    sensor_pose = pose_layer_eval(frame_index)
    sensor_position = sensor_pose[0:3, 3]
    sensor_rotation = sensor_pose[0:3, 0:3]
    sensor_rotation = rearrange(sensor_rotation, 'a b -> b a')

    # the masks depend only on the iteration the model was saved at, so they are fixed here
    pos_mask = barf_mask.get_barf_mask(current_iter=model_iteration, L=prep_config['pos_emb_dim'])
    dir_mask = barf_mask.get_barf_mask(current_iter=model_iteration, L=prep_config['dir_emb_dim'])

    points_eval_list = []
    for ray_direction_eval in ray_directions_eval_batchs:
        num_rays = ray_direction_eval.shape[0]

        ## coarse pass
        sampling_pos_coarse, sampling_dir_coarse, delta_coarse, magnitudes_coarse = get_sampling_pos_dir_eval(
            ray_direction_eval=ray_direction_eval,
            sensor_position=sensor_position,
            sensor_rotation=sensor_rotation,
            num_bins=num_bins_coarse,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device)

        emb_pos_coarse = SurrogateFourierFeature.apply(sampling_pos_coarse, prep_config['pos_emb_dim'])
        emb_dir_coarse = SurrogateFourierFeature.apply(sampling_dir_coarse, prep_config['dir_emb_dim'])
        with torch.no_grad():
            density_coarse = nerf_model_eval(emb_pos_coarse * pos_mask, emb_dir_coarse * dir_mask)

        # coarse alpha blending
        density_coarse = rearrange(density_coarse, '(n b) -> n b', n=num_rays, b=num_bins_coarse)
        alpha_pred_coarse = 1 - torch.exp(-density_coarse * delta_coarse)
        K = torch.cumprod((1 - alpha_pred_coarse), 1)   # K is just a temporary variable
        ones = torch.ones((num_rays, 1), device=device)
        T_pred_coarse = torch.cat((ones, K[:, :-1]), 1)  # Assume the cumulative transmittance of first sample along the ray is 1
        h_pred_coarse = T_pred_coarse*alpha_pred_coarse

        ## fine pass, sampled from where the coarse pass expects the surface
        # the first edge is extrapolated backwards by one bin width, and clamped at the near
        # plane so that sample_pdf() cannot place a fine sample behind the sensor
        first_bin_edge = magnitudes_coarse[:, 0] - (magnitudes_coarse[:, 1] - magnitudes_coarse[:, 0])
        first_bin_edge = torch.clamp(first_bin_edge, min=near_plane)
        bin_edges = torch.hstack((first_bin_edge.unsqueeze(1), magnitudes_coarse))
        magnitudes_fine = sample_pdf(bin_edges, h_pred_coarse, num_bins_fine=num_bins_coarse+num_bins_fine)
        sampling_pos_fine, sampling_dir_fine, deltas_fine = get_fine_sampling_pos_dir_eval(
            ray_direction_eval=ray_direction_eval,
            sensor_position=sensor_position,
            sensor_rotation=sensor_rotation,
            magnitudes_fine=magnitudes_fine,
            device=device)

        emb_pos_fine = SurrogateFourierFeature.apply(sampling_pos_fine, prep_config['pos_emb_dim'])
        emb_dir_fine = SurrogateFourierFeature.apply(sampling_dir_fine, prep_config['dir_emb_dim'])
        with torch.no_grad():
            density_fine = nerf_model_eval(emb_pos_fine * pos_mask, emb_dir_fine * dir_mask)
        density_fine = rearrange(density_fine, '(r b) -> r b', r=num_rays, b=num_bins_coarse+num_bins_fine)

        # fine alpha blending
        alpha_pred_fine = 1 - torch.exp(-density_fine * deltas_fine)
        K = torch.cumprod((1 - alpha_pred_fine), 1)   # K is just a temporary variable
        ones = torch.ones((num_rays, 1), device=device)
        T_pred_fine = torch.cat((ones, K[:, :-1]), 1)  # Assume the cumulative transmittance of first sample along the ray is 1
        h_pred_fine = T_pred_fine*alpha_pred_fine

        # take the distance where the termination distribution peaks, rather than its mean
        specific_h_pred = h_pred_fine / deltas_fine
        max_indices = torch.argmax(specific_h_pred, dim=1, keepdim=True)
        one_hot = torch.zeros_like(specific_h_pred)
        one_hot.scatter_(1, max_indices, 1)
        depth_eval = torch.sum(one_hot*magnitudes_fine, axis=1)

        # get mapped points with formula `p = o + d*r`
        ray_direction_eval_rotated = torch.matmul(ray_direction_eval, sensor_rotation)
        sensor_position_tiled = sensor_position.unsqueeze(0).expand(num_rays, 3)
        points_eval_batch = sensor_position_tiled + depth_eval.unsqueeze(1)*ray_direction_eval_rotated

        points_eval_list.append(points_eval_batch.cpu().numpy())

    return np.vstack(points_eval_list)


def render_all_views(nerf_model_eval, pose_layer_eval, data_dict, SurrogateFourierFeature, barf_mask,
                     prep_config: dict, model_iteration: int, num_sensors: int, num_bins: int,
                     near_plane: float, far_plane: float, device='cpu'):
    """Render every frame and stack the results into one cloud, still in cube units."""
    mapped_points_list = []
    for frame_index in range(num_sensors):
        mapped_points = render_a_view(nerf_model_eval=nerf_model_eval,
                                      pose_layer_eval=pose_layer_eval,
                                      data_dict=data_dict,
                                      SurrogateFourierFeature=SurrogateFourierFeature,
                                      barf_mask=barf_mask,
                                      prep_config=prep_config,
                                      model_iteration=model_iteration,
                                      num_bins=num_bins,
                                      frame_index=frame_index,
                                      near_plane=near_plane,
                                      far_plane=far_plane,
                                      device=device)
        mapped_points_list.append(mapped_points)
        # flushed so progress still shows up when the output is piped to a file
        print(f"rendered frame {frame_index + 1}/{num_sensors}, {len(mapped_points)} points", flush=True)

    combined_points = np.vstack(mapped_points_list)
    print(f"rendered {num_sensors} frames into {len(combined_points)} points")
    return combined_points


def save_world_map(points_cube: np.ndarray, scaling_factor: float, centre_point: np.ndarray,
                   out_path: Path, voxel_size_meter: float):
    """Take the finished map back to metres, downsample it once and write it as a binary .pcd.

    The downsampling is done here, over the whole map rather than per frame, so that the
    overlap between views is thinned the same way make_raw_map.py thins it.
    """
    points_world = denormalise_points_world_frame(points_cube, scaling_factor, centre_point)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size_meter)
    print(f"downsampled to {len(point_cloud.points)} points at {voxel_size_meter} m")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), point_cloud, write_ascii=False)
    print(f"map saved to {out_path}")
    return point_cloud


def main(nerf_model_path: Path, voxel_size_meter: float, num_bins: int):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"using {device} device")

    job_dir, config, dataset_dir, scaling_factor, centre_point = find_job_inputs(nerf_model_path)
    model_iteration = model_iteration_of(nerf_model_path)
    pose_layer_path = find_pose_checkpoint(job_dir, model_iteration)

    # the loader measures the normalisation again as it goes; the run's saved factors are the
    # ones the model was trained with, so a disagreement means the dataset moved since then
    data_dict, num_sensors, max_distance, measured_scaling_factor, _ = load_data(input_dir=dataset_dir,
                                                                                vox_size_meter=VOX_DOWN_METER)
    if abs(measured_scaling_factor - scaling_factor) > 1e-6:
        print(f"WARNING: dataset now measures a scaling factor of {measured_scaling_factor}, "
              f"but the run was trained with {scaling_factor} -- rendering with the run's value")

    # the sampling interval is bounded the same way training bounds it: the near plane comes
    # from the run's config, the far plane from the longest ray in the data
    near_plane = float(config['training']['near_plane'])
    far_plane = float(max_distance) + 0.1

    nerf_model_eval, pose_layer_eval, SurrogateFourierFeature, barf_mask = load_model_eval(
        nerf_model_path=nerf_model_path,
        pose_layer_path=pose_layer_path,
        prep_config=config['preperation'],
        num_sensors=num_sensors,
        device=device)

    # render in cube units, then take the finished map back to metres
    points_cube = render_all_views(nerf_model_eval=nerf_model_eval,
                                   pose_layer_eval=pose_layer_eval,
                                   data_dict=data_dict,
                                   SurrogateFourierFeature=SurrogateFourierFeature,
                                   barf_mask=barf_mask,
                                   prep_config=config['preperation'],
                                   model_iteration=model_iteration,
                                   num_sensors=num_sensors,
                                   num_bins=num_bins,
                                   near_plane=near_plane,
                                   far_plane=far_plane,
                                   device=device)

    # the map belongs to the checkpoint that rendered it, so name it after that iteration
    out_path = job_dir / 'rendered_maps' / f"iter{model_iteration}.pcd"
    save_world_map(points_cube, scaling_factor, centre_point, out_path, voxel_size_meter)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="render a map from a trained density field, one view per training frame")
    parser.add_argument("nerf_model_path", help="NeRF checkpoint written by a training run, at <job_dir>/nerf/<iteration>.pth", type=str)
    parser.add_argument("-v", "--voxel_size_meter", help="voxel downsampling size of the map, in metres",
                        type=float, default=VOXEL_SIZE_METER)
    parser.add_argument("-b", "--num_bins", help="coarse samples per ray, the fine pass adds as many again",
                        type=int, default=NUM_BINS)
    args = parser.parse_args()
    main(Path(args.nerf_model_path), args.voxel_size_meter, args.num_bins)
