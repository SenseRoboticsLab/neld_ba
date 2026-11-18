import argparse
import open3d as o3d
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud_path", help="path to the pose layer file", type=str, default=None)
    args = parser.parse_args()
    point_cloud_path = Path(args.point_cloud_path)
    pcd = o3d.io.read_point_cloud(str(args.point_cloud_path))
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([pcd], window_name = 'point cloud')
