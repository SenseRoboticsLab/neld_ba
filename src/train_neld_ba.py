# for parsing data
import toml
import csv
import shutil
import logging
import argparse
from collections import namedtuple
from types import SimpleNamespace
from pathlib import Path

# for training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# helpers
from training_helpers.load_helper import load_data
from training_helpers.dataset import RayShuffleSamplerDataset
from training_helpers.normalisation import save_normalisation
from training_helpers.nerf_models import lidarNerfDensity
from training_helpers.embeddings import SurrogateFourierFeature, BARFMasks
from training_helpers.poses_layers import PoseLayer
from training_helpers.volume_sampling import get_fine_sampling_pos_dir, get_sampling_pos_dir, sample_pdf
from training_helpers.alpha_blending_helper import render_rays, get_target_term_dist

# where a run writes its results; derived from the config rather than declared in it
OutputDirs = namedtuple('OutputDirs', ['root', 'nerf', 'pose', 'traj'])


def dict_to_namespace(d):
    '''Recursively convert the dictionary to a SimpleNamespace'''
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d


def setup_output_dirs(io_config: SimpleNamespace) -> OutputDirs:
    '''Create the output directory tree for this job and return it as an OutputDirs.

    Everything a run writes hangs off root -- the log, the metrics csv, the normalisation
    and the archived config sit directly in it, the checkpoints in the subdirectories.
    '''
    root = Path(io_config.output_dir) / io_config.job_name
    out_dirs = OutputDirs(root=root, nerf=root / 'nerf', pose=root / 'pose', traj=root / 'traj')
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    return out_dirs

def build_dataloader(data_dict: dict, batch_size: int):
    dataset = RayShuffleSamplerDataset(
        ray_distances=data_dict["ray_distance"], 
        ray_directions=data_dict["ray_direction"], 
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def build_models(prep_config: SimpleNamespace, data_dict: dict, num_sensors: int,
                 scaling_factor: float, centre_point, device='cpu'):
    '''Construct the nerf model, pose layer, surrogate fourier feature and barf mask.

    Returns (nerf_model, pose_layer, SurrogateFourierFeature, barf_mask, start_learn_pose_iter).
    '''
    ## nerf model
    nerf_model = lidarNerfDensity(
        embedding_dim_pos=prep_config.pos_emb_dim,
        embedding_dim_dir=prep_config.dir_emb_dim,
        hidden_dim=prep_config.hidden_dim,
        device=device,
    )

    ## pose layer
    init_r = torch.zeros(size=(num_sensors, 3), dtype=torch.float32)
    init_t = torch.zeros(size=(num_sensors, 3), dtype=torch.float32)
    for i in range(num_sensors):
        init_r[i] = torch.tensor(data_dict["sensor_rotation"][i])
        init_t[i] = torch.tensor(data_dict["sensor_position"][i])
    pose_layer = PoseLayer(
        timestamps=data_dict['timestamps'],
        num_cams=num_sensors,
        scaling_factor=scaling_factor,
        centre_point=centre_point,
        init_r=init_r,
        init_t=init_t, 
        device=device,
    )

    nerf_model.to(device = device)
    pose_layer.to(device = device)

    ## barf mask for gradual activation of spectrums
    barf_mask = BARFMasks(prep_config.start_barf_iter, prep_config.end_barf_iter, prep_config.pos_emb_dim, device)

    # delay pose learning
    start_learn_pose_iter = int(prep_config.end_barf_iter * (prep_config.start_learn_pose_L / prep_config.pos_emb_dim))

    return nerf_model, pose_layer, SurrogateFourierFeature, barf_mask, start_learn_pose_iter


def decay_gamma(start_lr: float, end_lr: float, num_iter: int, name: str) -> float:
    '''Per-iteration multiplier that carries start_lr to end_lr over num_iter iterations.
    '''
    if start_lr <= 0 or end_lr <= 0:
        raise ValueError(f"{name} learning rates must be positive, "
                         f"got start {start_lr} and end {end_lr}")
    if end_lr > start_lr:
        raise ValueError(f"{name} end learning rate {end_lr} is above its start {start_lr}, "
                         f"which would grow the learning rate instead of decaying it")
    if num_iter < 1:
        raise ValueError(f"num_iter must be at least 1 to schedule the {name} learning rate, "
                         f"got {num_iter}")
    return (end_lr / start_lr) ** (1.0 / num_iter)


def build_optimizers(prep_config: SimpleNamespace, num_iter: int, nerf_model, pose_layer):
    '''Construct the three Adam optimizers and their StepLR schedulers.
    Returns (optimizer_nerf, optimizer_rotation, optimizer_translation,
             scheduler_nerf, scheduler_rotation, scheduler_translation).
    '''
    # optimizers
    optimizer_nerf = torch.optim.Adam(nerf_model.parameters(),lr= prep_config.nerf_learning_rate)
    optimizer_rotation = torch.optim.Adam([pose_layer.r], lr=prep_config.pose_rotation_learning_rate)
    optimizer_translation = torch.optim.Adam([pose_layer.t], lr=prep_config.pose_translation_learning_rate)

    # schedules
    gamma_nerf = decay_gamma(prep_config.nerf_learning_rate,
                             prep_config.nerf_learning_rate_end, num_iter, 'nerf')
    gamma_rotation = decay_gamma(prep_config.pose_rotation_learning_rate,
                                 prep_config.pose_rotation_learning_rate_end, num_iter, 'pose rotation')
    gamma_translation = decay_gamma(prep_config.pose_translation_learning_rate,
                                    prep_config.pose_translation_learning_rate_end, num_iter, 'pose translation')
    logging.info(f"per-iteration learning rate decay over {num_iter} iterations: "
                 f"nerf {gamma_nerf:.9f}, rotation {gamma_rotation:.9f}, "
                 f"translation {gamma_translation:.9f}")

    scheduler_nerf = StepLR(optimizer_nerf, step_size=1, gamma=gamma_nerf)
    scheduler_rotation = StepLR(optimizer_rotation, step_size=1, gamma=gamma_rotation)
    scheduler_translation = StepLR(optimizer_translation, step_size=1, gamma=gamma_translation)

    return (optimizer_nerf, optimizer_rotation, optimizer_translation,
            scheduler_nerf, scheduler_rotation, scheduler_translation)


def archive_config(config_path: Path, out_dir: Path):
    '''Move the config that drove this run into the job directory
    '''
    config_path = Path(config_path)
    archived_path = out_dir / config_path.name
    if archived_path.resolve() == config_path.resolve():
        logging.info(f"config already lives in the job directory at {archived_path}")
        return archived_path

    shutil.move(str(config_path), str(archived_path))
    logging.info(f"config moved to {archived_path}")
    return archived_path


def setup_csv_logger(out_dir: Path, hier_sampling: bool):
    '''Write the training-metrics CSV header and return the csv file path.'''
    csv_file = out_dir / 'training_metrics.csv'
    csv_headers = ['iteration', 'loss_total', 'loss_h_coarse', 'loss_d_coarse',
                   'lr_nerf', 'lr_rotation', 'lr_translation', 'num_bins_coarse', 'spectrum']
    if hier_sampling:
        csv_headers.extend(['loss_h_fine', 'loss_d_fine'])
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    return csv_file


def run_batch(batch, nerf_model, pose_layer, fourier_feature, barf_mask,
              prep_config: SimpleNamespace, train_config: SimpleNamespace, iteration: int, device='cpu'):
    '''Run one batch forward pass and compute losses.
    '''
    KL_loss, l1_loss = nn.KLDivLoss(reduction = 'batchmean'), nn.L1Loss()

    # parse data and get pose
    ray_distances, ray_directions, sensor_ids = batch
    ray_distances = ray_distances.to(device)
    ray_directions = ray_directions.to(device)
    sensor_poses = torch.stack([pose_layer(int(id)) for id in sensor_ids])

    # IMPORTANT
    num_rays = ray_distances.shape[0]

    ## dynamic num_bins
    spectrum = barf_mask.get_current_spectrum(iteration, prep_config.pos_emb_dim)
    num_bins = int(train_config.num_coarse_bins_start + (train_config.num_coarse_bins_end - train_config.num_coarse_bins_start)*(spectrum / prep_config.pos_emb_dim))
    num_bins_fine = int(train_config.num_fine_bins_start + (train_config.num_fine_bins_end - train_config.num_fine_bins_start)*(spectrum / prep_config.pos_emb_dim))

    ## apply volume sampling
    sampling_positions_coarse, sampling_directions_coarse, deltas_coarse, magnitudes_coarse = get_sampling_pos_dir(sensor_poses=sensor_poses,
                                                                                        ray_direction=ray_directions,
                                                                                        near_plane=train_config.near_plane,
                                                                                        far_plane=train_config.far_plane,
                                                                                        num_bins=num_bins,
                                                                                        num_rays=num_rays,
                                                                                        device = device,)

    ## apply embeddings to sampled positions
    pos_mask = barf_mask.get_barf_mask(current_iter=iteration, L=prep_config.pos_emb_dim)
    dir_mask = barf_mask.get_barf_mask(current_iter=iteration, L=prep_config.dir_emb_dim)
    emb_pos_coarse = fourier_feature.apply(sampling_positions_coarse, prep_config.pos_emb_dim)
    emb_dir_coarse = fourier_feature.apply(sampling_directions_coarse, prep_config.dir_emb_dim)
    emb_pos_coarse_masked = emb_pos_coarse * pos_mask
    emb_dir_coarse_masked = emb_dir_coarse * dir_mask

    ## inference
    density_pred_coarse = nerf_model(emb_pos_coarse_masked, emb_dir_coarse_masked)
    alpha_pred_coarse = 1 - torch.exp(-density_pred_coarse * deltas_coarse)

    ## compute predicted cumulative transmittance from density prediction, and targets
    h_pred_coarse, d_pred_coarse = render_rays(alpha_pred=alpha_pred_coarse,
                                            magnitudes=magnitudes_coarse,
                                            num_rays=num_rays,
                                            num_bins=num_bins,
                                            device=device)

    ## conduct fine sampling
    if train_config.hier_sampling:
        # get bin edges, the first one extrapolated backwards by one bin width and clamped at
        first_bin_edge = magnitudes_coarse[:,0] - (magnitudes_coarse[:,1] - magnitudes_coarse[:,0])
        first_bin_edge = torch.clamp(first_bin_edge, min=train_config.near_plane)
        bin_edges = torch.hstack((first_bin_edge.unsqueeze(1), magnitudes_coarse))

        # get fine sampling positions, and combine with coarse samplign positions
        magnitudes_fine_temp = sample_pdf(bin_edges,h_pred_coarse, num_bins_fine).detach().clone()
        # magnitudes_fine_temp = magnitudes_fine_temp.detach().clone()    # doing this makes fine sampling useless for pose optimisation, very fucked up
        magnitudes_fine, _ = torch.sort(torch.cat((magnitudes_fine_temp, magnitudes_coarse), dim=1))
        sampling_positions_fine, sampling_directions_fine, deltas_fine = get_fine_sampling_pos_dir(magnitudes_fine=magnitudes_fine,
                                                                                                sensor_poses=sensor_poses,
                                                                                                ray_directions=ray_directions,
                                                                                                device=device)

        # apply embeddings to sampled positions
        emb_pos_fine = fourier_feature.apply(sampling_positions_fine, prep_config.pos_emb_dim)
        emb_dir_fine = fourier_feature.apply(sampling_directions_fine, prep_config.dir_emb_dim)
        emb_pos_fine_masked = emb_pos_fine * pos_mask
        emb_dir_fine_masked = emb_dir_fine * dir_mask

        ## fine inference
        density_pred_fine = nerf_model(emb_pos_fine_masked, emb_dir_fine_masked)
        alpha_pred_fine = 1 - torch.exp(-density_pred_fine * deltas_fine)

        ## compute predicted cumulative transmittance from density prediction, and targets
        h_pred_fine, d_pred_fine = render_rays(alpha_pred=alpha_pred_fine,
                                                magnitudes=magnitudes_fine,
                                                num_rays=num_rays,
                                                num_bins=num_bins+num_bins_fine,
                                                device=device)

    ## get targets
    h_target_coarse = get_target_term_dist(magnitude=magnitudes_coarse, ray_distance=ray_distances, num_rays=num_rays, device=device)
    if train_config.hier_sampling:
        h_target_fine = get_target_term_dist(magnitude=magnitudes_fine, ray_distance=ray_distances, num_rays=num_rays, device=device)
    d_target = ray_distances.squeeze_(1)        # shape: [num_rays]

    # compute losses
    # taking log for h_pred because pytorch kl_loss assume predicted value to be in log space
    loss_h_coarse = KL_loss(h_pred_coarse.log(), h_target_coarse)
    loss_d_coarse = l1_loss(d_pred_coarse, d_target)

    metrics = {'loss_h_coarse': loss_h_coarse, 
               'loss_d_coarse': loss_d_coarse,
               'num_bins': num_bins, 
               'spectrum': spectrum}

    if train_config.hier_sampling:
        loss_h_fine = KL_loss((h_pred_fine).log(), h_target_fine)
        loss_d_fine = l1_loss(d_pred_fine, d_target)
        loss_together = (train_config.lambda_coarse * (
                train_config.lambda_d * loss_d_coarse
                + train_config.lambda_h * loss_h_coarse
            ) + train_config.lambda_fine * (
                train_config.lambda_d * loss_d_fine
                + train_config.lambda_h * loss_h_fine
            )
        )
        metrics.update({'loss_h_fine': loss_h_fine, 'loss_d_fine': loss_d_fine})
    else:
        loss_together = (train_config.lambda_d * loss_d_coarse
                         + train_config.lambda_h_coarse * loss_h_coarse)

    return loss_together, metrics


def save_checkpoints(iteration, nerf_model, pose_layer, out_dirs: OutputDirs):
    '''Checkpoint the nerf model and the pose layer.
    '''
    # Save models
    try:
        nerf_out_path = out_dirs.nerf / f"iter{iteration}.pth"
        torch.save(nerf_model.state_dict(), nerf_out_path)
        logging.info(f"NeRF model saved to {nerf_out_path}")
    except Exception as e:
        logging.error(f"Error saving NeRF model at iteration {iteration}: {e}")

    # the .pth keeps the learnt state in cube units; save_as_tum writes the same poses in metres
    try:
        pose_out_path = out_dirs.pose / f"iter{iteration}.pth"
        traj_out_path = out_dirs.traj / f"iter{iteration}.txt"
        torch.save(pose_layer.state_dict(), pose_out_path)
        pose_layer.save_as_tum(traj_out_path)
        logging.info(f"Pose model saved to {pose_out_path}")
    except Exception as e:
        logging.error(f"Error saving pose model at iteration {iteration}: {e}")

def train(prep_config: SimpleNamespace, train_config: SimpleNamespace, io_config: SimpleNamespace,
          out_dirs: OutputDirs, device = 'cpu'):
    ### Load datas ###
    (data_dict, num_sensors, max_distance, 
     scaling_factor, centre_point) = load_data(io_config.input_dir, prep_config.vox_down_meter)

    logging.info("Loaded data.")
    train_config.far_plane = float(max_distance) + 0.1  # override far plane with the longest ray in the data

    ### Create models ###
    dataloader = build_dataloader(data_dict, prep_config.batch_size)

    nerf_model, pose_layer, fourier_feature, barf_mask, start_learn_pose_iter = build_models(
        prep_config=prep_config, 
        data_dict=data_dict, 
        num_sensors=num_sensors,
        scaling_factor=scaling_factor,
        centre_point=centre_point,
        device=device)

    # optimizers and schedulers
    (optimizer_nerf, optimizer_rotation, optimizer_translation,
     scheduler_nerf, scheduler_rotation, scheduler_translation) = build_optimizers(prep_config,
                                                                                   train_config.num_iter,
                                                                                   nerf_model, pose_layer)

    # save normalisation parameters for reference
    save_normalisation(out_dirs.root, scaling_factor, centre_point, num_sensors)

    ### Prepare for training ###
    start_learn_pose_flag = False
    end_learn_pose_flag= False
    csv_file = setup_csv_logger(out_dirs.root, train_config.hier_sampling)

    # get number of epoch based on number of total iteration specified in config
    num_epoch = -(-train_config.num_iter // len(dataloader))    # the -//- trick to round number up to nearest integer

    # IMPORTANT
    iteration = 0

    ### Start training ###
    print("Start training")
    for _ in range(num_epoch):
        for batch in dataloader:
            # unfreeze pose when spectrum is activated
            if start_learn_pose_flag is False and iteration >= start_learn_pose_iter:
                pose_layer.set_trainable(trainable_r=train_config.learn_rot, trainable_t=train_config.learn_trans)
                start_learn_pose_flag = True

            # freeze pose 
            if end_learn_pose_flag is False and iteration >= train_config.end_learn_pose_iter:
                pose_layer.set_trainable(trainable_r=False, trainable_t=False)
                end_learn_pose_flag = True

            ## back propergation
            optimizer_rotation.zero_grad()
            optimizer_translation.zero_grad()
            optimizer_nerf.zero_grad()

            # forward pass and losses
            loss_together, metrics = run_batch(batch, nerf_model, pose_layer, fourier_feature,
                                               barf_mask, prep_config, train_config, iteration, device)

            # back propergate
            loss_together.backward()
            optimizer_rotation.step()
            optimizer_translation.step()
            optimizer_nerf.step()

            # step scheduler every iteration
            scheduler_nerf.step()
            scheduler_rotation.step()
            scheduler_translation.step()

            # write losses to csv
            if iteration % 2000 == 0:
                csv_data = [iteration, 
                            loss_together.item(), 
                            metrics['loss_h_coarse'].item(),
                            metrics['loss_d_coarse'].item(),
                            optimizer_nerf.param_groups[0]['lr'],
                            optimizer_rotation.param_groups[0]['lr'],
                            optimizer_translation.param_groups[0]['lr'],
                            metrics['num_bins'], 
                            metrics['spectrum']]
                if train_config.hier_sampling:
                    csv_data.extend([metrics['loss_h_fine'].item(), metrics['loss_d_fine'].item()])
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_data)

            # checkpoints
            if iteration % io_config.checkpoint_interval_iter == 0 or iteration == (train_config.num_iter - 1):
                save_checkpoints(iteration=iteration,
                                 nerf_model=nerf_model,
                                 pose_layer=pose_layer,
                                 out_dirs=out_dirs)

            # exit training if all models have been saved
            if iteration >= train_config.num_iter:
                break

            # IMPORTANT
            iteration += 1

        # exit training after all iterations
        if iteration >= train_config.num_iter:
            break
    return

def main(args):
    config = dict_to_namespace(toml.load(args.input_config_path))

    # the run's output tree is made once here, before anything needs to write into it
    out_dirs = setup_output_dirs(config.data_io)

    # set up logging
    log_file = out_dirs.root / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    # get device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None and args.gpu >= 0 and args.gpu < torch.cuda.device_count() else 'cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using {device} device")

    # keep the config that drove this run next to its results
    archive_config(args.input_config_path, out_dirs.root)

    # train model
    train(
        prep_config=config.preperation,
        train_config=config.training,
        io_config=config.data_io,
        out_dirs=out_dirs,
        device=device,
    )
    logging.info("Finish training")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_config_path", help="path of configuration toml file", default = "./configs/template_config.toml", type = str)
    parser.add_argument("--gpu", help="Specify GPU ID to use (default: auto)", type=int, default=None)
    args = parser.parse_args()
    main(args)