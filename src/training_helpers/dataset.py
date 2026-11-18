import torch
import numpy as np
from torch.utils.data import Dataset

class RayShuffleSamplerDataset(Dataset):
    def __init__(self, ray_distances, ray_directions):
        """
        Initialize the dataset.

        Args:
            images (list of np.ndarray): List of 2D or 3D arrays representing images.
                                         Each image should be of the same shape.
        """
        self.ray_distances = torch.cat([torch.tensor(ray_distance, dtype=torch.float32) for ray_distance in ray_distances])
        self.ray_directions = torch.cat([torch.tensor(ray_direction, dtype=torch.float32) for ray_direction in ray_directions])
        self.sensor_id = torch.cat([torch.full((len(ray_distance),1),i) for i, ray_distance in enumerate(ray_directions)])
        self.num_sensors = len(ray_distances)

    def __len__(self):
        """
        Return the total number of samples.
        Since we randomly sample, len is treated as the number of possible iterations.
        """
        return len(self.ray_distances)

    def __getitem__(self, idx):
        """
        Retrieve a batch of samples using the shuffled indices.

        Args:
            idx (int): Index of the sample to retrieve (corresponds to batch index).

        Returns:
            tuple: (ray_distance_sampled, ray_direction_sampled, sensor_index)
        """
        # Fetch corresponding data
        ray_distance_sampled = self.ray_distances[idx]
        ray_direction_sampled = self.ray_directions[idx]
        sensor_id = self.sensor_id[idx]
        return ray_distance_sampled, ray_direction_sampled, sensor_id
