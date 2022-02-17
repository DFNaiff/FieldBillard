# -*- coding: utf-8 -*-

import torch

import fieldbillard


R = 1.0
n = 5
x = torch.rand(n)*2*R - R
y = torch.rand(n)*2*R - R
coupling = 1.0
charge = 1.0
mass = 1.0
square_density = 10.0
dt = 0.1

system = fieldbillard.NBodySystem(x, y, mass=mass, charge=charge, coupling=coupling)
square = fieldbillard.fields.Square(2*R, charge_density=square_density)
system.add_field_object(square)
for i in range(100):
    system.step(dt)