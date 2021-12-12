import torch
import torch.nn.functional as F
import torch.nn as nn

class STL10VGG(nn.Module):
    def __init__(self, batch_size = 100):
        super(STL10VGG, self).__init__()

        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(3, 96, 7, stride=2)

        self.conv2 = nn.Conv2d(96, 64, 5, stride=2)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=3)

        self.linear1 = nn.Linear(1152, 128)

        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = F.relu(out1)

        out2 = self.conv2(out1)
        out2 = F.relu(out2)

        out3 = self.conv3(out2)
        out3 = F.relu(out3)

        out3 = self.maxpool(out3)

        # out3 = out3.view(self.batch_size, -1)
        out3 = out3.view(-1, 1152)

        out4 = self.linear1(out3)
        out4 = F.relu(out4)

        out5 = self.linear2(out4)

        return out5
