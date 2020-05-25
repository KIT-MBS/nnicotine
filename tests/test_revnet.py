import torch
from pytest import approx
from torch import nn
from nnicotine.modules import ReversibleConvBlock, ReversibleSequential


def cos_angle(x1, x2):
    """
    computes the angle between two tensors, inputs are flattened and treated as vectors. Result in degrees.
    """
    with torch.no_grad():
        return (torch.dot(x1.flatten(), x2.flatten())/x1.norm()/x2.norm()).item()

    # return torch.acos(torch.dot(x1.flatten(), x2.flatten())/ (torch.norm(x1) * torch.norm(x2))).item() * 180./3.14159


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
    revnet = ReversibleConvBlock(3, 3, padding=1, dropout=0.)
    revnet.output_stack=[]
    x2 = torch.randn(2, 6, 10, 10, requires_grad=True)

    y1 = notrevnet(x1)
    y2 = revnet(x2)
    revnet.output_stack.append(y2.detach())

    with torch.no_grad():
        assert x1.eq(x2).all()
        assert y1.eq(y2).all()
        assert (notrevnet.f.weight == revnet.f.weight).all()

    y1.sum().backward()
    y2.sum().backward()

    with torch.no_grad():
        assert x2.grad.numpy() == approx(x1.grad.numpy())
        assert revnet.f.weight.grad.numpy() == approx(notrevnet.f.weight.grad.numpy(), rel=1e-5)
        assert cos_angle(revnet.f.weight.grad, notrevnet.f.weight.grad) == approx(1.)


def test_revnet():
    for n in range(2, 7):
        print(n)
        torch.manual_seed(s)
        notrevnet = nn.Sequential(*[NotReversibleBlock(3, 3, padding=1)]*n)
        x1 = torch.randn(2, 6, 10, 10, requires_grad=True)
        y1 = notrevnet(x1)

        torch.manual_seed(s)
        revnet = ReversibleSequential(*[ReversibleConvBlock(3, 3, padding=1, dropout=0.)]*n)
        x2 = torch.randn(2, 6, 10, 10, requires_grad=True)
        y2 = revnet(x2)

        with torch.no_grad():
            assert x1.eq(x2).all()
            assert y1.eq(y2).all()
            weights1 = [l.f.weight for l in notrevnet]
            weights2 = [l.f.weight for l in revnet]
            for w1, w2 in zip(weights1, weights2):
                assert (w1 == w2).all()

        y1.sum().backward()
        y2.sum().backward()

        with torch.no_grad():
            assert x2.grad.numpy() == approx(x2.grad.numpy())

            for w1, w2 in zip(weights1, weights2):
                print("largest absolute error:: ", (w1.grad - w2.grad).abs().max().item())
                print("largest relative error: ", ((w1.grad.abs() - w2.grad.abs())/w1.grad.abs()).max().item())
                assert w2.grad.numpy() == approx(w1.grad.numpy(), rel=5e-3)
                assert cos_angle(w1, w2) == approx(1.)

def test_rev_rng_retention():
    for n in range(2, 7):
        print(n)
        torch.manual_seed(s)
        notrevnet = nn.Sequential(*[NotReversibleBlock(3, 3, padding=1, dropout=0.5)]*n)
        x1 = torch.randn(2, 6, 10, 10, requires_grad=True)
        y1 = notrevnet(x1)

        torch.manual_seed(s)
        revnet = ReversibleSequential(*[ReversibleConvBlock(3, 3, padding=1, dropout=0.5)]*n)
        x2 = torch.randn(2, 6, 10, 10, requires_grad=True)
        y2 = revnet(x2)

        with torch.no_grad():
            assert x1.eq(x2).all()
            assert y1.eq(y2).all()
            weights1 = [l.f[0].weight for l in notrevnet]
            weights2 = [l.f[0].weight for l in revnet]
            for w1, w2 in zip(weights1, weights2):
                assert (w1 == w2).all()

        y1.sum().backward()
        y2.sum().backward()

        with torch.no_grad():
            assert x2.grad.numpy() == approx(x2.grad.numpy())

            for w1, w2 in zip(weights1, weights2):
                print("largest absolute error:: ", (w1.grad - w2.grad).abs().max().item())
                print("largest relative error: ", ((w1.grad.abs() - w2.grad.abs())/w1.grad.abs()).max().item())
                assert w2.grad.numpy() == approx(w1.grad.numpy(), rel=5e-3)
                assert cos_angle(w1, w2) == approx(1.)
    return

def test_rev_device_rng_retention():
    for n in range(2, 7):
        print(n)
        torch.manual_seed(s)
        notrevnet = nn.Sequential(*[NotReversibleBlock(3, 3, padding=1, dropout=0.5)]*n)
        notrevnet.to(device="cuda")

        x1 = torch.randn(2, 6, 10, 10, requires_grad=True, device="cuda")
        y1 = notrevnet(x1)

        torch.manual_seed(s)
        revnet = ReversibleSequential(*[ReversibleConvBlock(3, 3, padding=1, dropout=0.5)]*n)
        revnet.to(device="cuda")
        x2 = torch.randn(2, 6, 10, 10, requires_grad=True,device="cuda")
        y2 = revnet(x2)

        with torch.no_grad():
            assert x1.eq(x2).all()
            assert y1.eq(y2).all()
            weights1 = [l.f[0].weight for l in notrevnet]
            weights2 = [l.f[0].weight for l in revnet]
            for w1, w2 in zip(weights1, weights2):
                assert (w1 == w2).all()

        y1.sum().backward()
        y2.sum().backward()

        with torch.no_grad():
            assert x2.grad.cpu().numpy() == approx(x2.grad.cpu().numpy())

            for w1, w2 in zip(weights1, weights2):
                print("largest absolute error:: ", (w1.grad - w2.grad).abs().max().item())
                print("largest relative error: ", ((w1.grad.abs() - w2.grad.abs())/w1.grad.abs()).max().item())
                assert w2.grad.cpu().numpy() == approx(w1.grad.cpu().numpy(), rel=5e-3)
                assert cos_angle(w1, w2) == approx(1.)
    return

if __name__ == "__main__":
    test_revblock()
    test_revnet()
    test_rev_rng_retention()
    test_rev_device_rng_retention()

    print("all tests passed")
