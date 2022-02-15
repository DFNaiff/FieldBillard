# -*- coding: utf-8 -*-
import math
import collections

import torch

from . import system
from . import fields


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

        
def create_system_from_design(design, mass, charge):
    if design == "4-Diamond":
        x = torch.tensor([0.7, 0.0, -0.7, 0.0])
        y = torch.tensor([0.0, 0.7, 0.0, -0.7])
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
    elif design == "3-Random":
        x, y = _sample_uniform_unit_circle(3, 0.8)
    elif design == "4-Random":
        x, y = _sample_uniform_unit_circle(4, 0.8)
    elif design == "5-Random":
        x, y = _sample_uniform_unit_circle(5, 0.8)
    elif design == "6-Random":
        x, y = _sample_uniform_unit_circle(6, 0.8)
    elif design == "12-Random":
        x, y = _sample_uniform_unit_circle(12, 0.8)
    elif design == "24-Random":
        x, y = _sample_uniform_unit_circle(24, 0.8)
    syst = system.NBodySystem(x, y, mass=mass, charge=charge)
    return syst


def set_system_frame(syst, design, charge_density):
    if design == "Circle":
        obj = fields.Ring(1.0, charge_density=charge_density)
    elif design == "Square":
        obj = fields.Square(2.0, charge_density=charge_density)
    elif design == "Hash":
        obj = fields.Hash(2.0, charge_density=charge_density)
    syst.add_field_object(obj)
    

def set_integrator(syst, integrator):
    syst.set_integrator(integrator.lower())
    
    
def _sample_uniform_unit_circle(N, rmax=1.0):
    r = torch.sqrt(torch.rand(N))*rmax
    theta = torch.rand(N)*2*math.pi
    x = r*torch.cos(theta)
    y = r*torch.sin(theta)
    return x, y