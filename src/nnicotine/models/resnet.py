import torch
import torch.nn as nn
from ..modules import PositionalEncoding, OuterConcatenation

class Block(nn.Module):
    def __init__(self, channels, dilation, dropout=0.85):
        super(Block, self).__init__()

        self.n = nn.BatchNorm2d(channels)
        self.r = nn.ELU()
        self.c = nn.Conv2d(channels, channels, 3, dilation=dilation, padding=dilation)
        self.drop = nn.Dropout(p=0.85)

    def forward(self, x):
        x = self.n(x)

        return x + self.drop(self.c(self.r(x)))

# TODO fields
class ResNet(nn.Module):
    def __init__(self, num_classes=64):
        super(ResNet, self).__init__()
        embed_dim = 256

        self.embedding = nn.Embedding(21, embed_dim)
        self.positional = PositionalEncoding()
        self.outer_cat = OuterConcatenation()
        dilations = [1, 2, 4, 8]

        blocks = [Block(256, dilations[i%len(dilations)]) for i in range(7*4)]
        blocks.extend([nn.BatchNorm2d(256), nn.ELU(), nn.Conv2d(256, 128, 1)])
        # TODO should be 48*4 blocks
        blocks.extend([Block(128, dilations[i%len(dilations)]) for i in range(12*4)])

        self.blocks = nn.Sequential(*blocks)
        self.pre_blocks = nn.Sequential(nn.Conv2d(2*embed_dim + 2 + 20*20, 256, kernel_size=1))
        self.post_blocks = nn.Sequential(nn.ELU(), nn.Conv2d(128, num_classes, kernel_size=1))

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
