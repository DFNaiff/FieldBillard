# -*- coding: utf-8 -*-
import math
import collections

import torch
import numpy as np

from . import system
from . import fields
from . import integrators


class Memory(object):
    def __init__(self, alpha, alpha_lim=0.05):
        self.alpha = alpha
        self.alpha_lim = alpha_lim
        self.maxlen = int(math.log(alpha_lim)/math.log(alpha))
        self.deque = collections.deque([], maxlen=self.maxlen)
        
    def append(self, x):
        self.deque.append(x)
        
    def iterate_with_alpha(self):
        for i, item in enumerate(reversed(self.deque)):
            yield item, self.alpha**i

        
def create_system_from_design(design, noise, mass, charge, pradius, npoints, darwin_coupling):
    if design == "4-Diamond":
        x = torch.tensor([0.8, 0.0, -0.8, 0.0])
        y = torch.tensor([0.0, 0.8, 0.0, -0.8])
    elif design == "4-Square":
        x = torch.tensor([0.7, -0.7, -0.7, 0.7])
        y = torch.tensor([0.7, 0.7, -0.7, -0.7])
    elif design == "4-Cross":
        x = torch.tensor([0.3, 0.0, -0.3, 0.0])
        y = torch.tensor([0.0, 0.3, 0.0, -0.7])
    elif design == "3-Isosceles":
        x = torch.tensor([0.0, -0.3, 0.3])
        y = torch.tensor([0.7, -0.3, -0.3])
    elif design == "3-Isosceles-B":
        x = 0.8*torch.tensor([0.0, -math.sqrt(2)/2, math.sqrt(2)/2])
        y = 0.8*torch.tensor([1.0, -math.sqrt(2)/2, math.sqrt(2)/2])
    elif design == "3-Equilateral":
        x = 0.8*torch.tensor([0.0, math.cos(math.pi/6), -math.cos(math.pi/6)])
        y = 0.8*torch.tensor([1.0, -math.sin(math.pi/6), -math.sin(math.pi/6)])
    elif design == "N-Random-Circle":
        x, y = _sample_uniform_unit_circle(npoints, pradius)
    elif design == "N-Random-Square":
        x, y = _sample_uniform_unit_square(npoints, pradius)
    elif design == "N-Equilateral":
        x, y = _make_equilateral_designs(pradius, npoints)
    x += noise*torch.randn_like(x)
    y += noise*torch.randn_like(y)
    syst = system.NBodySystem(x, y, mass=mass, charge=charge,
                              darwin_coupling=darwin_coupling)
    return syst


def set_system_frame(syst, design, charge_density):
    if design == "Circle":
        obj = fields.Ring(1.0, charge_density=charge_density)
        syst.add_field_object(obj)
    elif design == "Square":
        obj = fields.Square(2.0, charge_density=charge_density)
        syst.add_field_object(obj)
    elif design == "Hash":
        obj = fields.Hash(2.0, charge_density=charge_density)
        syst.add_field_object(obj)
    elif design == "Periodic":
        syst.points.lx = 2.0
        syst.points.ly = 2.0
    elif design == "XPeriodic":
        syst.points.lx = 2.0
        obj1 = fields.HorizontalLine(-1.0, charge_density=charge_density)
        obj2 = fields.HorizontalLine(1.0, charge_density=charge_density)
        syst.add_field_object(obj1)
        syst.add_field_object(obj2)
    elif design == "YPeriodic":
        syst.points.ly = 2.0
        obj1 = fields.VerticalLine(-1.0, charge_density=charge_density)
        obj2 = fields.VerticalLine(1.0, charge_density=charge_density)
        syst.add_field_object(obj1)
        syst.add_field_object(obj2)
    else:
        raise ValueError

def set_fixed_points(syst, design, N, charge):
    if design == "None": #Edge case
        return np.array([])[..., None], np.array([])[..., None]
    else:
        if design == "RandomCircle":
            x, y = _sample_uniform_unit_circle(N)
        elif design == "RandomSquare":
            x = torch.rand(N)*2 - 1
            y = torch.rand(N)*2 - 1
        else:
            raise ValueError
        if syst.points.periodic:
            obj = fields.PeriodicFixedPoints(x, y, charge,
                                             syst.points.lx,
                                             syst.points.ly,
                                             syst.points.cx,
                                             syst.points.cy)
        else:
            obj = fields.FixedPoints(x, y, charge)
        syst.add_field_object(obj)
    return np.array(x), np.array(y)

def set_integrator(syst, integrator):
    syst.set_integrator(integrator.lower())
    
    
def _sample_uniform_unit_circle(N, rmax=1.0):
    r = torch.sqrt(torch.rand(N))*rmax
    theta = torch.rand(N)*2*math.pi
    x = r*torch.cos(theta)
    y = r*torch.sin(theta)
    return x, y


def _sample_uniform_unit_square(N, rmax=1.0):
    x = (torch.rand(N)*2 - 1)*rmax
    y = (torch.rand(N)*2 - 1)*rmax
    return x, y


def _make_equilateral_designs(r, N):
    theta = torch.linspace(0, 2*math.pi, N+1)[:-1] + math.pi/2
    x = r*torch.cos(theta)
    y = r*torch.sin(theta)
    return x, y