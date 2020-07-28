import torch
from torch import nn


class PixelGenerator(nn.Module):

    """ Generate RGB colour pixels that DO NOT occur frequently in the real H&E distribution 
    
    Attributes:
        main (nn.Sequential): maps from a latent variable z to an offset vector
    """

    def __init__(self):
        super().__init__()

        
        self.main = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, n):
        """ forward pass of the pixel generator module

        Args:
            n (int): number of pixels to generate/sample
        """

        # sample n latent variables (each of these are considered to be an RGB colour pixel)
        z = torch.rand(n, 3).to(self.main[0].weight.device)

        # generator observes z and estimates offset vectors
        offset = self.main(z)

        # apply offset to z
        out = z + offset

        # ensure the updated colours are in the range [0., 1.]
        out = out.clamp(min=0.0, max=1.0)

        return out, offset
