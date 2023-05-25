______________________________________________________________________

<div align="center">

# GAN Implementation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

My implementation of GAN, DCGAN, cGAN.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/phamgialinhlx/GANs
cd GANs

# [OPTIONAL] create conda environment
conda create -n gan python=3.8 -y
conda activate gan

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train_gan.py trainer=cpu

# train on GPU (default)
python src/train_gan.py 
```

You can override any parameter from command line like this

```bash
# train on FashionMNIST dataset
python src/train_gan.py datamodule=fashion_mnist
# use cGAN model
python src/train_gan.py model=cgan
# use DCGAN model with small architecture (image size: 28 x 28)
python src/train_gan.py model=small_dcgan 
```
