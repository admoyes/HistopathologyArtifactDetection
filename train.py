import torch
from torch.utils.data import DataLoader
from torchvision import utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import toml
config = toml.load("./config.toml")

from Models import train_artifact_detector
from Data import PatchDataset


""" [0] dataset and dataloader """
INCIDENT_LIGHT = torch.FloatTensor(config["data"]["incident_light"]).to(torch.device(config["device"]))

# check if a custom tansform is needed
if bool(config["data"]["use_custom_transform"]):
    from Data import CustomTransform as TrainingTransform
else:
    from Data import DefaultTransform as TrainingTransform

# build the training dataset
training_dataset = PatchDataset(config["training"]["path_to_patches"], TrainingTransform())

# build the training data loader
training_dataloader = DataLoader(
    training_dataset,
    batch_size=int(config["training"]["batch_size"]),
    shuffle=True,
    num_workers=int(config["training"]["num_workers"]),
    drop_last=True
)

""" [1] Training """
for epoch, gan, data in train_artifact_detector(
    training_dataloader,
    config["path_to_state"],
    bool(config["training"]["load_state"]),
    INCIDENT_LIGHT,
    int(config["training"]["epochs"]),
    float(config["training"]["learning_rate"]),
    int(config["training"]["log_interval"]),
    config["device"]
    ):

    # sample some fake pixels and plot them
    with torch.no_grad():
        pixels_fake, _ = gan.generator(5000)
        pixels_fake = pixels_fake.detach().cpu()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        pixels_fake[:, 0],
        pixels_fake[:, 1],
        pixels_fake[:, 2],
        c=pixels_fake
    )
    fig.savefig("./output/training/fake_pixels.png")
    plt.close(fig)

    # apply the classifier to data and plot them
    with torch.no_grad():
        batch_size, _, patch_size, _ = data.size()
        pred_real = gan.classifier(data.permute(0, 2, 3, 1).contiguous().view(-1, 3))
        pred_real = pred_real.view(batch_size, patch_size, patch_size).unsqueeze(1)
        
    utils.save_image(data.detach(), "./output/training/data.png", normalize=False)
    utils.save_image(pred_real.detach(), "./output/training/pred_data.png", normalize=False)

    # save state
    torch.save(gan.state_dict(), config["path_to_state"])