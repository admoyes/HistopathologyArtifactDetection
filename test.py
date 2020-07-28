import torch
from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import toml
config = toml.load("./config.toml")

from Models import PixelGAN
from Data import PatchDataset


""" [0] dataset and dataloader """
INCIDENT_LIGHT = torch.FloatTensor(config["data"]["incident_light"]).to(torch.device(config["device"]))

# check if a custom tansform is needed
if bool(config["data"]["use_custom_transform"]):
    from Data import CustomTransform as TestingTransform
else:
    from Data import DefaultTransform as TestingTransform

# build the testing dataset
testing_dataset = PatchDataset(config["testing"]["path_to_patches"], TestingTransform())

# build the testing data loader
testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=int(config["testing"]["batch_size"]),
    shuffle=True,
    num_workers=int(config["testing"]["num_workers"]),
    drop_last=True
)

""" [1] build model and load state """
gan = PixelGAN().to(torch.device(config["device"]))
gan.load_state_dict(torch.load(config["path_to_state"]))

count = 0
for batch_id, data in enumerate(testing_dataloader):

    data = data.to(torch.device(config["device"]))

    # normalise data by incident light
    data = data / INCIDENT_LIGHT.view(1, 3, 1, 1)

    # compute probability scores for data
    with torch.no_grad():
        batch_size, _, patch_size, _ = data.size()
        pred_real = gan.classifier(data.permute(0, 2, 3, 1).contiguous().view(-1, 3))
        pred_real = pred_real.view(batch_size, patch_size, patch_size).unsqueeze(1)
        
    # save images and predictions
    for i in range(batch_size):
        data_i = (data[i, ...].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pred_i = (pred_real[i, 0, ...].cpu().numpy() * 255).astype(np.uint8)

        Image.fromarray(data_i).save("./output/testing/data/{}.png".format(count))
        Image.fromarray(pred_i).save("./output/testing/pred/{}.png".format(count))

        count += 1