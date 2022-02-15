# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "../..")

import torch

import badnbody


R = 1.0
n = 5
x = torch.rand(n)*2*R - R
y = torch.rand(n)*2*R - R
coupling = 1.0
charge = 1.0
mass = 1.0
square_density = 10.0

system = badnbody.NBodySystem(x, y, mass=mass, charge=charge, coupling=coupling)
square = badnbody.fields.Square(2*R, charge_density=square_density)
system.add_field_object(square)