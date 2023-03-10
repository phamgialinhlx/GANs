''' fid code and inception model from https://github.com/mseitzer/pytorch-fid '''

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from scipy import linalg
from .inception import InceptionV3 # https://github.com/mseitzer/pytorch-fid
import pickle
import torch
import numpy as np
from tqdm import tqdm

def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False).eval()

    return inception_feat


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    ''' https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py '''
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

class FIDCallback(pl.callbacks.base.Callback):
    '''
    db_stats - pickle file with inception stats on real data
    z_sampler - function to sample generator input
    fid_name - name for logging
    n_samples - number of samples for FID
    '''
    def __init__(self, db_stats, z_sampler, fid_name, n_samples=5000, batch_size=16, eval_every=10000):
        self.z_sampler = z_sampler
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.fid_name = fid_name

        # Load inception statistics computed on real data
        with open(db_stats, 'rb') as f:
            embeds = pickle.load(f)
            self.real_mean = embeds['mean']
            self.real_cov = embeds['cov']

    def to(self, device):
        self.inception = self.inception.to(device)
        self.z_samples = [z.to(device) for z in self.z_samples]

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        '''
        Initialize random noise and inception module
        I keep the model and the noise on CPU when it's not needed to preserve memory; could also be initialized on pl_module.device
        '''
        self.z_samples = [self.z_sampler(self.batch_size, device=torch.device('cpu')) for i in range(0, self.n_samples, self.batch_size)]
        self.inception = load_patched_inception_v3()
        #self.inception = self.inception.to(pl_module.device)
        print('FID initialized')
        # Keeping last global step so that the code is not run more than once in case when we use accumulating gradients; perhaps there's a better way in newer PL version
        self.last_global_step = trainer.global_step - 1

    @rank_zero_only
    def on_batch_start(self, trainer, pl_module):
        if (trainer.global_step + 1) % self.eval_every != 0 or trainer.global_step == self.last_global_step: # + 1
            return

        pl_module.eval()
        
        with torch.no_grad():
            self.to(pl_module.device)
            features = []
            
            for i, z in enumerate(self.z_samples):
                inputs = z
                fake = pl_module(z) # get fake images
                feat = self.inception(fake)[0].view(fake.shape[0], -1) # compute features
                features.append(feat.to('cpu'))

            features = torch.cat(features, 0)[:self.n_samples].numpy()

            sample_mean = np.mean(features, 0)
            sample_cov = np.cov(features, rowvar=False)

            fid = calc_fid(sample_mean, sample_cov, self.real_mean, self.real_cov)

            # log FID
            for logger in pl_module.logger:
                logger.log_metrics({self.fid_name: fid}, step=trainer.global_step)
            self.to(torch.device('cpu'))

        pl_module.train()
        self.last_global_step = trainer.global_step