from typing import Any, List

import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import hydra

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

class GAN(LightningModule):

    def __init__(
        self,
        gen: torch.nn.Module,
        disc: torch.nn.Module,
        optimizer: torch.nn.Module,
        z_dim: int = 10,
        device='cpu'
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['gen', 'disc'])

        self.gen = gen
        self.disc = disc

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def get_disc_loss(self, real, num_images):
        '''
        Return the loss of the discriminator given inputs.
        Parameters:
            gen: the generator model, which returns an image given z-dimensional noise
            disc: the discriminator model, which returns a single-dimensional prediction of real/fake
            criterion: the loss function, which should be used to compare 
                the discriminator's predictions to the ground truth reality of the images 
                (e.g. fake = 0, real = 1)
            real: a batch of real images
            num_images: the number of images the generator should produce, 
                    which is also the length of the real images
            z_dim: the dimension of the noise vector, a scalar
            device: the device type
        Returns:
            disc_loss: a torch scalar loss value for the current batch
        '''
        # Create noise vectors and generate a batch (num_images) of fake images. 
        noise = get_noise(num_images, self.hparams.z_dim, device=self.device)
        fake = self.gen(noise)

        # Get the discriminator's prediction of the fake image 
        # and calculate the loss.
        fake_pred = self.disc(fake.detach())
        
        # Get the discriminator's prediction of the real image and calculate the loss.
        real_pred = self.disc(real)

        # Calculate the discriminator's loss by averaging the real 
        # and fake loss and set it to disc_loss.
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2

        return disc_loss

    def get_gen_loss(self, num_images): 
        '''
        Return the loss of the generator given inputs.
        Parameters:
            gen: the generator model, which returns an image given z-dimensional noise
            disc: the discriminator model, which returns a single-dimensional prediction of real/fake
            criterion: the loss function, which should be used to compare 
                the discriminator's predictions to the ground truth reality of the images 
                (e.g. fake = 0, real = 1)
            num_images: the number of images the generator should produce, 
                    which is also the length of the real images
            z_dim: the dimension of the noise vector, a scalar
            device: the device type
        Returns:
            gen_loss: a torch scalar loss value for the current batch
        '''
        # Create noise vectors and generate a batch of fake images. 
        noise = get_noise(num_images, self.hparams.z_dim, device=self.device)
        fake = self.gen(noise)

        # Get the discriminator's prediction of the fake image.
        fake_pred = self.disc(fake)
        # Calculate the generator's loss.
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        return gen_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Flatten the batch of real images from the dataset
        real = batch[0].view(len(batch[0]), -1)

        if (optimizer_idx == 0):
            disc_loss = self.get_disc_loss(real, len(real))
            self.log("disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)
            return disc_loss
        else:
            gen_loss = self.get_gen_loss(len(real))
            self.log("gen_loss", gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        opt_g = self.hparams.optimizer(self.gen.parameters())
        opt_d = self.hparams.optimizer(self.disc.parameters())

        return [{"optimizer": opt_d}, {"optimizer": opt_g}]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gan_mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
