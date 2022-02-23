# -*- coding: utf-8 -*-
import math

import torch


def to_polar(x, y):
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta


def trapquad(f, a, b, N, *args):
    x = torch.linspace(a, b, N+1) #(n,)
    dx = (b - a)/N
    y = f(x, *args) #(n, *argdims)
    value = 0.5*dx*torch.sum((y[1:, ...] + y[:-1, ...]), axis=0) #(*argdims)
    return value


def circle_phi(r, N=100):
    def integrand(theta, r):
        if not isinstance(r, float):
            theta = theta.view(*(theta.shape + (1,)*theta.ndim))
        return 1/torch.sqrt(1 + r**2 - 2*r*torch.cos(theta))
    integral = trapquad(integrand, 0, 2*math.pi, N, r)
    integral = torch.nan_to_num(integral, 0.0)
    return integral


def upper_mask(N):
    return torch.triu(torch.ones(N, N) * float('inf'))


def diagonal_mask(N):
    return torch.diag(torch.ones(N) * float('inf'))


def wrap_from_center(tensor, lengths, centers):
    #x -> (x - center) + l;2 -> 
    #     (x - center + l/2)%l
    #  -> (x - center + l/2)%l - l/2
    tensor = tensor.clone()
    for i, length in enumerate(lengths):
        if length is None:
            continue
        center = centers[i]
        dislocation = length/2 - center
        tensor[..., i] = (tensor[..., i] + dislocation)%length - \
                          dislocation
    return tensor
    
    
    