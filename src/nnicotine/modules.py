import torch
from torch import nn
from .functional import reversible_block_forward


class LSHSelfAttention(nn.Module):
    def __init__(self):
        super(LSHAttention, self).__init__()

    def forward(self, x):
        return


class ChunkedFeedForward(nn.Module):
    def __init__(self):
        super(ChunkedFeedForward, self).__init__()

    def forward(self, x):
        return


class ReversibleBlock(nn.Module):
    # TODO check that it is contained in a reversible container or has access to an output stack in some other way?
    def __init__(self, f, g):
        super(ReversibleBlock, self).__init__()
        self.f = f
        self.g = g
        self.output_stack = None

    def forward(self, x):
        # NOTE input channel dim has to be divisible by two
        if self.output_stack is None:
            raise ValueError("output_stack of {} has to be set to a stack shared between reversible layers.".format(self))
        return reversible_block_forward(self.f, self.g, self.output_stack, x, preserve_rng_state=True, dim=1)


class ReversibleSequential(nn.Sequential):
    def __init__(self, *args, output_stack=[]):
        super(ReversibleSequential, self).__init__(*args)
        self.output_stack = output_stack
        # TODO first block should not put its input on the stack
        for module in self:
            assert isinstance(module, ReversibleBlock)
            module.output_stack = self.output_stack
        return

    def forward(self, x):
        with torch.no_grad():
            y = super(ReversibleSequential, self).forward(x)
            self.output_stack.append(y)
            return y


# TODO norm
class ReversibleConvBlock(ReversibleBlock):
    def __init__(self, channels, kernel_size, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', alpha=1.):
        conv = nn.Conv2d(
                channels, channels, kernel_size,
                stride=1, padding=padding, dilation=dilation,
                groups=groups, bias=bias, padding_mode=padding_mode
            )
        nonl = nn.ELU(alpha=alpha)
        super(ReversibleConvBlock, self).__init__(conv, nonl)


# TODO norm
# TODO LSH
class ReversibleLSHSelfAttentionBlock(ReversibleBlock):
    def __init__(self):
        attn = LSHSelfAttention()
        # TODO chunked ff
        ff = nn.Sequential()
        super(ReversibleLSHSelfAttentionBlock, self).__init__(attn, ff)
    def forward(self, x):
        return


class ReversibleCompressedAttentionBlock(ReversibleBlock):
    def __init__(self):
        return

    def forward(self):
        return


class SelfAttention(nn.Module):
    def __init__(self):
        return

    def forward(self, x):
        return
