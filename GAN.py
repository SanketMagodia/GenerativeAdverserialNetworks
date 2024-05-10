import torch.nn as nn
import hyperParams as params

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(params.latent_size, 100),
            nn.ReLU(),
            nn.Linear(100, params.image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(params.image_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)