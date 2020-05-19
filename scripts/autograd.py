import torch
from torch import nn
from torch.autograd import Function

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, x, m, b):
        print(ctx)
        with torch.no_grad():
            y = x.mm(m.t())
            y += b.unsqueeze(0).expand_as(y)
            ctx.save_for_backward(x, m, b)
        return y

    @staticmethod
    def backward(ctx, dy):
        print(ctx)
        x, m, b = ctx.saved_tensors
        dx = dy.mm(m)
        dm = dy.t().mm(x)
        db = dy.sum(0)
        return dx, dm, db

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.m = torch.randn(out_features, in_features, requires_grad=True)
        self.b = torch.randn(out_features, requires_grad=True)
    def forward(self, x):
        return LinearFunction.apply(x, self.m, self.b)


N, d_in, d_h, d_out = 4, 20, 30, 10

x = torch.randn(N, d_in, dtype=torch.float)
y = torch.randn(N, d_out, dtype=torch.float)

l1 = LinearLayer(d_in, d_h)
l2 = LinearLayer(d_h, d_out)

y_pred = l2(nn.functional.relu(l1(x)))

e = (y_pred - y).pow(2).sum()

e.backward()
