import torch
from torch import nn


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.gen_layer0 = self.make_gen_block(z_dim, hidden_dim * 16, kernel_size=4, stride=1, padding=0)
        self.gen_layer1 = self.make_gen_block(hidden_dim * 16, hidden_dim * 8, kernel_size=4, stride=2)
        self.gen_layer2 = self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2)
        self.gen_layer3 = self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2)
        self.gen_layer4 = self.make_gen_block(hidden_dim * 2, im_chan, kernel_size=4, final_layer=True)


    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh()                
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        # x = self.unsqueeze_noise(noise)
        out = self.gen_layer0(noise)
        out = self.gen_layer1(out)
        out = self.gen_layer2(out)
        out = self.gen_layer3(out)
        out = self.gen_layer4(out)
        
        return out

if __name__ == "__main__":
    z_dim = 50
    gen = Generator(z_dim=z_dim)
    noise = torch.rand(128, z_dim, 1, 1)
    print(noise.shape)
    tmp = gen.gen_layer0(noise)
    print(tmp.shape)
    tmp = gen.gen_layer1(tmp)
    print(tmp.shape)
    tmp = gen.gen_layer2(tmp)
    print(tmp.shape)
    tmp = gen.gen_layer3(tmp)
    print(tmp.shape)
    tmp = gen.gen_layer4(tmp)
    print(tmp.shape)
