import torch
import torch.nn as nn

class BaseLidarNerf(nn.Module):
    def __init__(self, embedding_dim_pos, embedding_dim_dir, hidden_dim, device = 'cpu'):

        super(BaseLidarNerf, self).__init__()
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_dir = embedding_dim_dir
        self.device = device

        # Shared blocks
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3 + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 + 1)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_dir * 6 + 3 + hidden_dim // 2, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Activation functions
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos, dir):
        raise NotImplementedError("Subclasses must implement the forward method.")

# just in case I need some other nerf models
class lidarNerfDensity(BaseLidarNerf):
    def __init__(self, embedding_dim_pos, embedding_dim_dir, hidden_dim, device = 'cpu'):
        super().__init__(
            embedding_dim_pos=embedding_dim_pos, 
            embedding_dim_dir=embedding_dim_dir, 
            hidden_dim=hidden_dim, 
            device=device
        )

    def forward(self, emb_x_masked, emb_d_masked):
        # Inference
        temp_0 = self.block1(emb_x_masked)
        temp_1 = torch.cat((temp_0, emb_x_masked), dim=1)  # Add skip connection
        temp_2 = self.block2(temp_1)
        temp_3, density = temp_2[:, :-1], self.softplus(temp_2[:, -1])  # Apply softplus for density
        # temp_4 = torch.cat((temp_3, emb_d_masked), dim=1)  # Add laser direction
        # reflectivity = self.sigmoid(self.block3(temp_4))  # Apply sigmoid for reflectivity
        return density 
