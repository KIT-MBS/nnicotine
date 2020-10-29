import torch
import torch.nn as nn
from ..modules import PositionalEncoding, OuterConcatenation

class Block(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Block, self).__init__()
        # TODO dilation
        # TODO dropout
        # TODO normalization
        # TODO topology

        self.n = nn.BatchNorm2d(channels)
        self.r1 = nn.ELU()
        self.c1 = nn.Conv2d(channels, channels, kernel_size, padding=int(kernel_size/2))

        return

    def forward(self, x):
        x = self.n(x)

        return x + self.c1(self.r1(x))

# TODO fields
class ResNet(nn.Module):
    def __init__(self, d=64, num_blocks=8, num_classes=64):
        super(ResNet, self).__init__()
        self.embed_dim = 30

        self.embedding = nn.Embedding(21, self.embed_dim)
        self.positional = PositionalEncoding()
        self.outer_cat = OuterConcatenation()
        blocks = [Block(d, 3) for i in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.pre_blocks = nn.Sequential(nn.Conv2d(2*self.embed_dim + 2 + 20*20, d, kernel_size=1)) # TODO norm
        self.post_blocks = nn.Sequential(nn.ELU(), nn.Conv2d(d, num_classes, kernel_size=1))
        return

    def forward(self, x):
        seq = x['sequence']
        couplings =  x['couplings']

        couplings = couplings.transpose(1, -1)

        y = self.embedding(seq)
        # NOTE channel first
        y = torch.transpose(y, 1, -1).contiguous()

        y = self.positional(y)

        y = self.outer_cat(y)
        y = torch.cat((y, couplings), 1)

        y = self.pre_blocks(y)
        y = self.blocks(y)
        y = self.post_blocks(y)

        return y
