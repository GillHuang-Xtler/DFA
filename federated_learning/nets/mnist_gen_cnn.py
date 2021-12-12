import torch
import torch.nn as nn

class NetGenBase(nn.Module):

    def __init__(self, z_dim):
        super(NetGenBase, self).__init__()
        self.z_dim = z_dim
        self.decoder = None
        self.fc_dinput = None

    def decode(self, z) -> torch.Tensor:
        pass

    def forward(self, z):
        return self.decode(z)

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.z_dim)
        samples = self.decode(z)
        return samples

    def generate(self, z):
        return self.decode(z)

class NetGenMnist(NetGenBase):

    def __init__(self, z_dim=128):
        super(NetGenMnist, self).__init__(z_dim)

        dim = 5 * 4 * 4
        self.fc_dinput = nn.Linear(self.z_dim, dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 10, 5, stride=1),  # 5*4*4=>10*8*8
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(10, 10, 5, stride=4),  # 10*8*8=>10*33*33
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10, 1, 6, stride=1),  # 10*33*33=>1*28*28
            nn.BatchNorm2d(1),
            # nn.Tanh(),  # the value range (-1, 1)
            nn.ReLU(),
        )

    def decode(self, z) -> torch.Tensor:
        x = self.fc_dinput(z)
        x = x.view(x.shape[0], 5, 4, 4)
        # print("x:", x.shape)
        x = self.decoder(x)

        # x = torch.sign(x)
        # x = torch.relu(x)
        return x
