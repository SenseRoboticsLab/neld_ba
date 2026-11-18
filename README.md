# NeLD-BA: Neural LiDAR Bundle Adjustment

[![arXiv](https://img.shields.io/badge/arXiv-2607.04169-b31b1b.svg)](https://arxiv.org/abs/2607.04169)

Official implementation of **Neural LiDAR Bundle Adjustment**.

Chin Yung Anson Hon, Kaicheng Zhang, Sen Wang — Imperial College London

[[Paper]](https://arxiv.org/abs/2607.04169)

## Overview

NeLD-BA jointly optimises a LiDAR neural density field and the sensor poses of a set of LiDAR scans. It starts from initial poses, for example from FAST-LIO2, and returns refined poses and a dense map. Our main observation is that, in LiDAR NeRFs, where the volume samples are placed along each ray matters a lot. NeLD-BA therefore:

- puts the fine samples just in front of the coarse samples, so they land on the sharp free-space-to-surface transition;
- clamps each ray's sampling interval to the normalised unit cube `[-1, 1]^3`, so long, sparse LiDAR rays are not wasted on empty space;
- uses a coarse-to-fine (BARF-style) positional-encoding schedule with a surrogate gradient, and starts optimising the poses once enough frequency bands are active;
- trains with a range L1 loss and a ray-termination-distribution loss.

On the Newer College and FusionPortable datasets, NeLD-BA reaches state-of-the-art accuracy in multi-view point-cloud registration and 3D mapping. See the [paper](https://arxiv.org/abs/2607.04169) for details.

## Quick Start

Requirements: Linux x86_64, an NVIDIA GPU with CUDA 12.x drivers, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo-url>
cd neld_ba
bash scripts/setup.bash
source .venv/bin/activate
```

The repository includes a small sample from Newer College `quad_hard` (20 scans) for testing.

```bash
python src/train_neld_ba.py configs/nc_quad_hard_mini.toml
```

Results go to `~/output/nc_quad_hard_mini/`. 

## Building Maps

**Raw map.** Register every input scan with a learnt trajectory and merge the scans:

```bash
python src/make_raw_map.py output/<job_name>/traj/iter<N>.txt [-v 0.05]
# writes output/<job_name>/raw_map/iter<N>.pcd
```

**Rendered map.** Re-render every LiDAR ray with the trained density field and keep one point per ray:

```bash
python src/make_rendered_map.py output/<job_name>/nerf/iter<N>.pth [-v 0.05] [-b 256]
# writes output/<job_name>/rendered_maps/iter<N>.pcd
```

`-v` sets the voxel downsampling size in metres, and `-b` sets the number of coarse and fine samples per ray. Both maps are in metres, so they can be compared directly with a ground-truth map. To view a map, run:

```bash
python utility/peek_pcd.py output/<job_name>/raw_map/iter<N>.pcd
```
## Data Format

Each sequence is a directory with one point cloud per scan and a trajectory of initial poses:

```
<data_root>/
  pcd/
    1625135731_975675136.pcd     # one scan, in the sensor frame, in metres
    ...
  poses.txt                      # initial poses, TUM format, in metres
```

- `poses.txt` has one pose per line, in [TUM format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats): `timestamp x y z qx qy qz qw`.
- Each `.pcd` file is named after its scan's timestamp. Files are matched to poses by the digits of the timestamp, so `1625135731_975675136.pcd` matches the pose with timestamp `1625135731.975675136`.

To check that the scans and poses line up before training, run:

```bash
python utility/peek_training_data.py <data_root>
```

## Configuration

All hyperparameters live in a TOML file; see [`configs/default.toml`](configs/default.toml) for the full list with comments.
## Outputs

```
output/<job_name>/
  <config>.toml              # the config used for this run
  normalisation.json         # scale and centre used to map metres into the unit cube
  training_metrics.csv       # losses over training
  nerf/iter<N>.pth           # NeRF checkpoints
  pose/iter<N>.pth           # pose-layer checkpoints
  traj/iter<N>.txt           # refined trajectory, TUM format, in metres
```

The trajectories in `traj/` can be evaluated directly with tools such as [evo](https://github.com/MichaelGrupp/evo).


## Datasets

The paper evaluates on:

- [Newer College](https://ori-drs.github.io/newer-college-dataset/): `cloister`, `math_easy`, `math_hard`, `quad_easy`, `quad_mid`, `quad_hard`
- [FusionPortable](https://fusionportable.github.io/dataset/fusionportable/): `canteen_day`, `canteen_night`, `garden_day`, `garden_night`

The initial poses come from FAST-LIO2. The ground truth is the survey-grade maps and trajectories released with each dataset.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{hon2026neldba,
  title         = {Neural LiDAR Bundle Adjustment},
  author        = {Hon, Chin Yung Anson and Zhang, Kaicheng and Wang, Sen},
  year          = {2026},
  eprint        = {2607.04169},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2607.04169}
}
```

## Acknowledgements

This code builds on ideas from [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF) and LiDAR NeRF methods. The sample data comes from the [Newer College dataset](https://ori-drs.github.io/newer-college-dataset/).

## License

The code in this repository is released under the [MIT License](LICENSE).