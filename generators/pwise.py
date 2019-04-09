import torch
from torch.nn import functional as F
import torchvision
import random



class Piecewise(torch.nn.Module):
    def __init__(self, params):
        super(Piecewise,self).__init__()
        # define number of filters
        pow = params['fpow'] if 'fpow' in params else 4
        nf = 2**pow
        # define enhancement module
        basis_param = params['n']
        self.emodule = PiecewiseBasis(basis_param)
        # define network
        self.c2 = torch.nn.Conv2d(   3,   nf, kernel_size=3, stride=2, padding=0)
        self.r2 = torch.nn.PReLU(num_parameters=  nf, init=0.25)
        self.c3 = torch.nn.Conv2d(  nf, 2*nf, kernel_size=3, stride=2, padding=0)
        self.r3 = torch.nn.PReLU(num_parameters=2*nf, init=0.25)
        self.c4 = torch.nn.Conv2d(2*nf, 4*nf, kernel_size=3, stride=2, padding=0)
        self.r4 = torch.nn.PReLU(num_parameters=4*nf, init=0.25)
        self.downsample = torch.nn.AvgPool2d(11, stride=1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.PReLU(num_parameters=64, init=0.25),
            torch.nn.Linear(64, self.emodule.parameters_count)
        )

    def forward(self, image, applyto=None):
        x = image
        if (image.size(2), image.size(3)) != (100, 100):
            x = _bilinear(x, 100, 100)
        x = x - 0.5
        x = self.r2(self.c2(x))
        x = self.r3(self.c3(x))
        x = self.r4(self.c4(x))
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        if not self.training:
            result = torch.clamp(result, 0, 1)
        return result


class EnhancementModule(torch.nn.Module):
    def __init__(self, parameters_count):
        super(EnhancementModule,self).__init__()
        self.parameters_count = parameters_count

    def forward(self, image, parameters):
        return image


class FunctionBasis(EnhancementModule):
    def __init__(self, basis_dimension):
        super(FunctionBasis,self).__init__(basis_dimension * 3)
        self.bdim = basis_dimension

    def expand(self, x):
        """Bx3xHxW -> Bx3xDxHxW  where D is the dimension of the basis."""
        raise NotImplemented

    def forward(self, image, parameters):
        x = self.expand(image)
        w = parameters.view(parameters.size(0), 3, -1)
        return torch.einsum("bcfij,bcf->bcij", (x, w))


class PiecewiseBasis(FunctionBasis):
    def __init__(self, dim):
        super(PiecewiseBasis,self).__init__(dim)
        nodes = torch.arange(dim).view(1, 1, -1, 1, 1).float()
        self.register_buffer("nodes", nodes)

    def expand(self, x):
        x = x.unsqueeze(2)
        return F.relu(1 - torch.abs((self.bdim - 1) * x - self.nodes))


def _bilinear(im, height, width):
    xg = torch.linspace(-1, 1, width, device=im.device)
    yg = torch.linspace(-1, 1, height, device=im.device)
    mesh = torch.meshgrid([yg, xg])
    grid = torch.stack(mesh[::-1], 2).unsqueeze(0)
    grid = grid.expand(im.size(0), height, width, 2)
    return F.grid_sample(im, grid)




if __name__ == "__main__":
    params = {
        "n":10
    }
    net = Piecewise(params)
    import ipdb; ipdb.set_trace()
    print(net)
    x = torch.rand(7, 3, 100, 100)
    y = net(x)
    print(y.size())
