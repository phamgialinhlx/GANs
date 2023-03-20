import torch
from torch import nn 

class Discriminator(nn.Module):
    def __init__(self, im_chan=11, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc_layer0 = self.make_disc_block(im_chan, hidden_dim)
        self.disc_layer1 = self.make_disc_block(hidden_dim, hidden_dim * 2)
        self.disc_layer2 = self.make_disc_block(hidden_dim * 2, 1, final_layer=True)

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, image):
        out = self.disc_layer0(image)
        out = self.disc_layer1(out)
        out = self.disc_layer2(out)

        return out.view(len(out), -1)

if __name__ == "__main__":
    disc = Discriminator(hidden_dim=64)
    print(disc)
    noise = torch.rand(128, 11, 28 , 28)
    print(noise.shape)
    tmp = disc.disc_layer0(noise)
    print(tmp.shape)
    tmp = disc.disc_layer1(tmp)
    print(tmp.shape)
    tmp = disc.disc_layer2(tmp)
    print(tmp.shape)
