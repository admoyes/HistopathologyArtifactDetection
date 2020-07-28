import torch
from torch import nn


class Classifier(nn.Module):

    """ Map observed colours to probability scores. These probability scores represent the likelihood of the observed colour belonging to the real H+E distribution

    Attributes:
        main (nn.Sequential): maps from colours to probability scores

    """

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(3, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)