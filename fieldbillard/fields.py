# -*- coding: utf-8 -*-

from . import utils

import torch


class FieldObject(object):
    def potential(x, y, charge, coupling):
        pass


class Ring(FieldObject):
    def __init__(self, radius, charge_density=1.0, x0=0.0, y0=0.0):
        super().__init__()
        self.radius = radius
        self.charge_density = charge_density
        self.x0 = x0
        self.y0 = y0
        
    def potential(self, x, y, charge, coupling=1.0):
        r, _ = utils.to_polar(x - self.x0, y - self.y0)
        value = coupling*self.charge_density*charge*utils.circle_phi(r/self.radius)
        return value


class HorizontalLine(FieldObject):
    def __init__(self, y0, charge_density=1.0):
        self.charge_density = charge_density
        self.y0 = y0
        
    def potential(self, x, y, charge, coupling=1.0):
        value = -coupling*self.charge_density*charge*utils.torch.log(torch.abs(y - self.y0))
        return value


class VerticalLine(FieldObject):
    def __init__(self, x0, charge_density=1.0):
        self.charge_density = charge_density
        self.x0 = x0
        
    def potential(self, x, y, charge, coupling=1.0):
        value = -coupling*self.charge_density*charge*torch.log(torch.abs(x - self.x0))
        return value
    

class Hash(FieldObject):
    def __init__(self, l, charge_density=1.0, x0=0.0, y0=0.0):
        self.l = l
        self.charge_density = charge_density
        self.x0 = x0
        self.y0 = y0
        self._set_lines()
    
    def potential(self, x, y, charge, coupling=1.0):
        value = self._upper.potential(x, y, charge, coupling) + \
                self._lower.potential(x, y, charge, coupling) + \
                self._left.potential(x, y, charge, coupling) + \
                self._right.potential(x, y, charge, coupling)
        return value
    
    def _set_lines(self):
        self._upper = HorizontalLine(self.y0 + self.l/2, self.charge_density)
        self._lower = HorizontalLine(self.y0 - self.l/2, self.charge_density)
        self._left = VerticalLine(self.x0 + self.l/2, self.charge_density)
        self._right = VerticalLine(self.x0 - self.l/2, self.charge_density)
        
        
class HorizontalFiniteLine(FieldObject):
    def __init__(self, y0, l, x0=0.0, charge_density=1.0):
        self.charge_density = charge_density
        self.y0 = y0
        self.x0 = x0
        self.l = l
        
    def potential(self, x, y, charge, coupling=1.0):
        dx, dy = x - self.x0, y - self.y0
        integral = torch.arcsinh((2*dx + self.l)/(2*torch.abs(dy))) - \
                   torch.arcsinh((2*dx - self.l)/(2*torch.abs(dy)))
        value = coupling*self.charge_density*charge*integral
        return value


class VerticalFiniteLine(FieldObject):
    def __init__(self, x0, l, y0=0, charge_density=1.0):
        self.charge_density = charge_density
        self.x0 = x0
        self.l = l
        self.y0 = y0
        
    def potential(self, x, y, charge, coupling=1.0):
        dx, dy = x - self.x0, y - self.y0
        integral = torch.arcsinh((2*dy + self.l)/(2*torch.abs(dx))) - \
                   torch.arcsinh((2*dy - self.l)/(2*torch.abs(dx)))
        value = coupling*self.charge_density*charge*integral
        return value


class Square(FieldObject):
    def __init__(self, l, charge_density=1.0, x0=0.0, y0=0.0):
        self.l = l
        self.charge_density = charge_density
        self.x0 = x0
        self.y0 = y0
        self._set_lines()
    
    def potential(self, x, y, charge, coupling=1.0):
        value = self._upper.potential(x, y, charge, coupling) + \
                self._lower.potential(x, y, charge, coupling) + \
                self._left.potential(x, y, charge, coupling) + \
                self._right.potential(x, y, charge, coupling)
        return value
    
    def _set_lines(self):
        self._upper = HorizontalFiniteLine(self.y0 + self.l/2, self.l, self.x0,
                                           self.charge_density)
        self._lower = HorizontalFiniteLine(self.y0 - self.l/2, self.l, self.x0,
                                           self.charge_density)
        self._left = VerticalFiniteLine(self.x0 + self.l/2, self.l, self.y0,
                                        self.charge_density)
        self._right = VerticalFiniteLine(self.x0 - self.l/2, self.l, self.y0,
                                         self.charge_density)


class FixedPoints(FieldObject):
    def __init__(self, x0, y0, charge=1.0):
        super().__init__()
        self.x0 = x0 #(n, )
        self.y0 = y0 #(n, )
        self._assert_positions()
        #Assert blablabla
        self.charge = charge #(n, )

    def potential(self, x, y, charge, coupling=1.0):
        d = torch.sqrt((x - self.x0)**2 + (y - self.y0)**2) #(n, )
        values = coupling*self.charge*charge/d
        value = torch.sum(values)
        return value