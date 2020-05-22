import torch

from torch.autograd import Function
from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states

# TODO rename dim parameter to be more descriptive
# TODO split into half residuals?
class ReversibleBlockFunction(Function):
    @staticmethod
    def forward(ctx, f, g, output_stack, x, preserve_rng_state, dim):
        if x.requires_grad == False: raise ValueError("Input to a reversible layer has to require grad.")
        check_backward_validity(x)
        ctx.preserve_rng_state = preserve_rng_state
        ctx.split_dim = dim
        ctx.output_stack = output_stack

        ctx.f = f
        ctx.g = g
        # NOTE that only the device of input is checked for rng state preservation not the forward functions (f and g) (see torch.utils.checkpoint)"

        with torch.no_grad():
            x1, x2 = torch.chunk(x, 2, dim=dim)
            if x1.size(dim) != x2.size(dim):
                raise ValueError("Input to Reversible Blocks have to be even in size along dim {}.".format(dim))
            ctx.had_cuda_in_fwd = torch.cuda.is_initialized()
            if preserve_rng_state:
                ctx.pre_f_cpu_state = torch.get_rng_state()
                if ctx.had_cuda_in_fwd:
                    ctx.pre_f_devices, ctx.pre_f_gpu_states = get_device_states(x)
            z = x1 + f(x2)
            if preserve_rng_state:
                ctx.pre_g_cpu_state = torch.get_rng_state()
                if ctx.had_cuda_in_fwd:
                    ctx.pre_g_devices, ctx.pre_g_gpu_states = get_device_states()
            y2 = x2 + g(z)
            y1 = z
            y = torch.cat((y1, y2), dim=dim)
        return y

    @staticmethod
    def backward(ctx, dy):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("ReversibleBlock is not compatible with .grad(), please use .backward() if possible (see torch.utils.checkpoint)")

        dim = ctx.split_dim

        x1 = None
        x2 = None
        f = ctx.f
        g = ctx.g
        with torch.no_grad():
            dy1, dy2 = torch.chunk(dy, 2, dim=dim)
            y = ctx.output_stack.pop()
            y1, y2 = torch.chunk(y, 2, dim=dim)

            z = y1

            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.pre_g_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.pre_g_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.pre_g_devices, ctx.pre_g_gpu_states)
                x2 = y2 - g(z)
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.pre_f_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.pre_f_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.pre_f_devices, ctx.pre_f_gpu_states)
                x1 = z - f(x2)
        x1.requires_grad = True
        x2.requires_grad = True

        # TODO check x1, x2 are leaves
        with torch.enable_grad():
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.pre_f_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.pre_f_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.pre_f_devices, ctx.pre_f_gpu_states)
                z = x1 + f(x2)
                y2 = x2 + g(z)
                y1 = z
                y = torch.cat((y1, y2), dim=dim)
        torch.autograd.backward(y, dy)
        dx1 = x1.grad
        dx2 = x2.grad
        if x1.grad is None or x2.grad is None: raise RuntimeError("Input gradients could not be computed")

        x = torch.cat((x1, x2), dim=dim)
        dx = torch.cat((dx1, dx2), dim=dim)

        ctx.output_stack.append(x)
        # NOTE has to be one out for each forward in
        return None, None, None, dx, None, None

def reversible_block_forward(f, g, output_stack, x, preserve_rng_state=True, dim=1):
    return ReversibleBlockFunction.apply(f, g, output_stack, x, preserve_rng_state, dim)
