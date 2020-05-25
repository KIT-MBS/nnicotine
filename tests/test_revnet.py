import torch
from pytest import approx
from torch import nn
from nnicotine.modules import ReversibleConvBlock


class NotReversibleBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding=0, dropout=0.):
        super(NotReversibleBlock, self).__init__()
        if dropout>0.:
            self.f = nn.Sequential(nn.Conv2d(channels, channels, kernel_size, padding=padding), nn.Dropout(p=dropout))
        else:
            self.f = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.g = nn.ELU(alpha=1.)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        z = x1 + self.f(x2)
        y2 = x2 + self.g(z)
        y1 = z
        return torch.cat((y1, y2), dim=1)


s = 42

def test_revblock():

    torch.manual_seed(s)
    notrevnet = NotReversibleBlock(3, 3, padding=1)
    x1 = torch.randn(2, 6, 10, 10, requires_grad=True)

    torch.manual_seed(s)
    revnet = ReversibleConvBlock(3, 3, padding=1)
    revnet.output_stack=[]
    x2 = torch.randn(2, 6, 10, 10, requires_grad=True)

    y1 = notrevnet(x1)
    y2 = revnet(x2)
    revnet.output_stack.append(y2.detach())

    assert x1.eq(x2).all()
    assert y1.eq(y2).all()
    assert (notrevnet.f.weight == revnet.f.weight).all()

    y1.sum().backward()
    y2.sum().backward()

    assert (x1.grad - x2.grad).max().item() == approx(0., abs=1e-6)
    assert (notrevnet.f.weight.grad - revnet.f.weight.grad).max().item() == approx(0., abs=1e-5)


def test_revnet():
    return

def test_rev_rng_retention():
    return

def test_rev_device_rng_retention():
    return

if __name__ == "__main__":
    test_revblock()
    test_revnet()
    test_rev_rng_retention()
    test_rev_device_rng_retention()

    print("all tests passed")