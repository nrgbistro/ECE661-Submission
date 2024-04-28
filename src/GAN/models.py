import torch.nn as nn
import numpy as np

class Generator(nn.Module):

    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size

        self.image_shape = (3,32,32)
        self.model = nn.Sequential(
                        nn.Linear(latent_size,128,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(128,256,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(256,512,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(512,1024,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(1024,2048,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(2048,int(np.prod(self.image_shape))),
                        nn.Tanh()
                    )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.image_shape)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.image_shape = (3,32,32)
        self.model = nn.Sequential(
                            nn.Linear(int(np.prod(self.image_shape)), 1024),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(1024,512),
                            nn.LeakyReLU(0.2,inplace=True),
                            nn.Linear(512, 256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(256, 1),
                            nn.Sigmoid(),
                    )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        output = self.model(img_flat)
        return output