import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
  
        self.disc_layer0 = self.make_disc_block(im_chan, hidden_dim * 2)
        self.disc_layer1 = self.make_disc_block(hidden_dim * 2, hidden_dim * 4)
        self.disc_layer2 = self.make_disc_block(hidden_dim * 4, hidden_dim * 8)
        self.disc_layer3 = self.make_disc_block(hidden_dim * 8, hidden_dim * 16)
        self.disc_layer4 = self.make_disc_block(hidden_dim * 16, 1, stride=1, padding=0, final_layer=True)

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Sigmoid()
            )

    def forward(self, image):
        out = self.disc_layer0(image)
        out = self.disc_layer1(out)
        out = self.disc_layer2(out)
        out = self.disc_layer3(out)
        out = self.disc_layer4(out)
        return out

if __name__ == "__main__":
    disc = Discriminator(hidden_dim=64)
    noise = torch.rand(128, 1, 64, 64)
    print(noise.shape)
    tmp = disc.disc_layer0(noise)
    print(tmp.shape)
    tmp = disc.disc_layer1(tmp)
    print(tmp.shape)
    tmp = disc.disc_layer2(tmp)
    print(tmp.shape)
    tmp = disc.disc_layer3(tmp)
    print(tmp.shape)
    tmp = disc.disc_layer4(tmp)
    print(tmp.shape)
