import torch
from torch import nn
from .classifier import Classifier
from .generator import PixelGenerator


class PixelGAN(nn.Module):

    def __init__(self):
        super().__init__()

        self.classifier = Classifier()
        self.generator = PixelGenerator()

    def train_classifier(self, x, n=1000):

        # reshape x
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        # sample n points from x
        idx = torch.randperm(x.size(0))
        x_sample = x[idx, :]

        # compute probability scores for x
        pred_real = self.classifier(x_sample)

        # sample non-H+E pixels from the generator
        with torch.no_grad():
            pixels_fake, _ = self.generator(n)

        pred_fake = self.classifier(pixels_fake)

        # calculate losses
        loss_real = torch.pow(pred_real - 1, 2).mean()
        loss_fake = torch.pow(pred_fake, 2).mean()

        loss = 0.5 * (loss_real + loss_fake)

        return loss

    def train_generator(self, n=1000):

        pixels_fake, offsets = self.generator(n)

        pred_fake = self.classifier(pixels_fake)

        loss = 0.5 * torch.pow(pred_fake, 2).mean() + torch.norm(offsets, dim=1).mean()

        return loss