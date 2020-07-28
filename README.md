# Histopathology Artifact Detection
This repository contains a PyTorch implementation of the artifact detection model proposed in [this paper](https://link.springer.com/chapter/10.1007/978-3-030-52791-4_18). This paper was presented at [MIUA2020](https://miua2020.com/) and won the *best paper award*. A video summary of this paper can be found [here](https://www.youtube.com/watch?v=HQ6kL6eVdig).

The artifact detection model using a sparse pixel generator uses a neural network inspired by generative adversarial networks to learn a decision boundary between colours that do frequently occur in H+E images and those colours which do not. A Sparse Pixel Generator (SPG) is defined which learns to generate pixels which *do not* frequently occur in the H+E distribution. A classifier is trained on real pixels sampled from H+E images as well as pixels sampled from the SPG model. Probability scores are assigned to colours and those which do not frequently occur in H+E images are assigned a low score. Thresholding can be applied to the scores to detect artifacts.

## Getting Started
- Our local environment is a typical conda environment
- Make sure you have a proper nvidia driver if using a GPU
- Install [PyTorch](https://pytorch.org/get-started/locally/) along with `torchvision` and `cudatoolkit` (if using GPU)
- Install `pillow`, `toml`, `tqdm` and `matplotlib`
	- Specific versions can be found in the `requirements.txt` file

### Training
- Ensure the `path_to_patches` parameter in `config.toml` under `training_data` points to a directory that is populated with tissue image patches
	- The model was originally trained with 128x128 pixel patches but this is not hard coded (See config)
- Run `python main.py`
- Model output will be output at `./output/training/`
- States will be saved under `.cache/gan.state`

### Testing
- Again, ensure the `path_to_patches` parameter in `config.toml` under `testing_data` is populated with tissue image patches.
- Run `python test.py`
- Model output will be saved to disk at `./output/testing/`
    - The input images will be saved under `./output/testing/data/`
    - The probability scores will be saved under `./output/testing/pred/`
    - No further processing is applied to the probability scores. The necessary thresholding value to accurately detect/segment artifacts will likely vary depending on the dataset and so this is left up to the user.
- Configuration is done in `config.toml`
- To test the model, run `python test.py`

### Recommendations
- An accurate estimation of the incident light is needed to properly perform stain separation.
    - This can be done by manually selecting pixels that **do not** contain tissue in your preferred image editing software (e.g. [GIMP](https://www.gimp.org/)) and looking at the *histogram* section which will let you see the average RGB intensity values. 
    - Once you have these values, make sure the `incident_light` parameter under `data` in the `config.toml` file is updated accordingly.

## Citation
If you use this code in your research, please cite the [paper](https://link.springer.com/chapter/10.1007/978-3-030-52791-4_18):

```
@inproceedings{moyes2020unsupervised,
  title={Unsupervised Deep Learning for Stain Separation and Artifact Detection in Histopathology Images},
  author={Moyes, Andrew and Zhang, Kun and Ji, Ming and Zhou, Huiyu and Crookes, Danny},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={221--234},
  year={2020},
  organization={Springer}
}
```
