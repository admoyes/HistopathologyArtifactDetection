import torch
from torch import optim
from .pixel_gan import PixelGAN
import os
from tqdm import tqdm


def train_artifact_detector(dataloader, state_path, load_state, incident_light, epochs, learning_rate, log_interval, device):

    gan = PixelGAN().to(torch.device(device))

    if load_state and len(state_path) > 0 and os.path.exists(state_path):
        gan.load_state_dict(torch.load(state_path))

    optim_classifier = optim.Adam(gan.classifier.parameters(), lr=learning_rate)
    optim_generator = optim.Adam(gan.generator.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        for batch_id, data in enumerate(dataloader):

            data = data.to(torch.device(device))
            data = data / incident_light.view(1, 3, 1, 1)

            # train the classifier
            optim_classifier.zero_grad()
            clf_loss = gan.train_classifier(data)
            clf_loss.backward()
            optim_classifier.step()

            # train the generator
            optim_generator.zero_grad()
            gen_loss = gan.train_generator()
            gen_loss.backward()
            optim_generator.step()            

        if epoch % log_interval == 0:
            yield epoch, gan, data