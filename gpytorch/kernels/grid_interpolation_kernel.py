import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.utils.interpolation import Interpolation
from gpytorch.lazy import ToeplitzLazyVariable


class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module, grid_size):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module

        grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)

        grid_diff = grid[1] - grid[0]

        self.grid_size = grid_size + 2
        self.grid = Variable(torch.linspace(0 - grid_diff, 1 + grid_diff, grid_size + 2))

    def forward(self, x1, x2, **kwargs):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            raise RuntimeError(' '.join([
                'The grid interpolation kernel can only be applied to inputs of a single dimension at this time \
                until Kronecker structure is implemented.'
            ]))

        x1 = (x1 - x1.min(0)[0].expand_as(x1)) / (x1.max(0)[0] - x1.min(0)[0]).expand_as(x1)
        x2 = (x2 - x2.min(0)[0].expand_as(x2)) / (x2.max(0)[0] - x2.min(0)[0]).expand_as(x2)

        J1, C1 = Interpolation().interpolate(self.grid.data, x1.data.squeeze())
        J2, C2 = Interpolation().interpolate(self.grid.data, x2.data.squeeze())

        k_UU = self.base_kernel_module(self.grid[0], self.grid, **kwargs).squeeze()

        K_XX = ToeplitzLazyVariable(k_UU, self.grid, J1, C1, J2, C2)

        return K_XX
