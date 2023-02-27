import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from models.dcgan.model import DCGAN

from src import utils
import numpy as np

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_gan_mnist.yaml")
def inference(cfg: DictConfig):
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer)


    gen = model.gen
    gen.eval()
    noise = torch.randn(1, cfg.model.z_dim)
    image = gen(noise)
    # print(image)
    image = np.reshape(image.detach().numpy(), [28, 28])
    plt.imsave("./abc.jpg", image)
    # plt.show()
    
if __name__ == "__main__":
    inference()