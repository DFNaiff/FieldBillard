# -*- coding: utf-8 -*-

from . import fields
from . import points
from . import integrators


class NBodySystem(object):
    def __init__(self, x, y, px=None, py=None, mass=1.0, charge=1.0,
                 integrator='leapfrog', coupling=1.0):
        self.points = points.MovingPoints(x, y, px, py, mass, charge)
        self.objects = []
        self.integrator = integrators.get_integrator(integrator)
        self.coupling = coupling
        
    def add_field_object(self, field_obj):
        assert isinstance(field_obj, fields.FieldObject)
        self.objects.append(field_obj)
    
    def step(self, dt):
        self.integrator(dt, self.points, self.objects, self.coupling)
        
    def set_integrator(self, integrator):
        self.integrator = integrators.get_integrator(integrator)