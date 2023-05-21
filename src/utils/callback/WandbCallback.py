import torch
import wandb
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import pytorch_lightning as pl
from typing import Any
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2
from src.utils.gan_utils import get_noise, get_one_hot_labels, combine_vectors

class WandbCallback(Callback):
    def __init__(self, z_dim:int = 64, batch_size=128, im_chan:int = 1, n_classes:int = 10, img_size:int = 28, use_fixed_noise:bool = True):
        self.size = [im_chan, img_size, img_size]
        # self.size = (1, 28, 28)
        self.z_dim = z_dim #z_dim
        self.batch_size = batch_size
        self.step = 500
        self.n_classes = n_classes
        self.last_batch = None
        self.fid = FrechetInceptionDistance(feature=64)
        self.use_fixed_noise = use_fixed_noise
        self.fake_noise = get_noise(self.batch_size, self.z_dim, device = 'cuda') # get noise

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if (len(self.last_batch) == 2):
            last_batch, labels = self.last_batch
        else:
            last_batch = self.last_batch

        if (not self.use_fixed_noise):
            self.fake_noise = get_noise(len(last_batch), self.z_dim, device = 'cuda')

        one_hot_labels = get_one_hot_labels(labels, self.n_classes)
        noise_and_labels = combine_vectors(self.fake_noise, one_hot_labels)

        fake = trainer.model(noise_and_labels)
        fake = fake.detach()
        
        image_unflat = fake.detach().cpu().view(-1, * self.size)
        image_grid = make_grid(image_unflat[:25], nrow=5)

        real_unflat = last_batch.view(-1, * self.size)
        real_grid = make_grid(real_unflat[:25], nrow=5)

        # logging using WandB
        wandb_logger = trainer.logger 
        wandb_logger.log_image(key='Samples', images=[wandb.Image(image_grid), wandb.Image(real_grid)], caption=["fake", "real"])
        

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        self.last_batch = batch
