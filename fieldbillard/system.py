# -*- coding: utf-8 -*-
from typing import Optional

import torch

from . import fields
from . import points
from . import integrators


class NBodySystem(object):
    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 px: Optional[torch.Tensor] = None,
                 py: Optional[torch.Tensor] = None,
                 mass: float = 1.0, charge:float = 1.0,
                 integrator: str ='sympleticverlet', coupling:float = 1.0,
                 darwin_coupling: Optional[float] = None, 
                 lx: Optional[float] = None, ly: Optional[float] = None,
                 cx: Optional[float] = 0.0, cy: Optional[float] = 0.0):
        """
        

        Parameters
        ----------
        x : torch.Tensor
            position x-coordinate.
        y : torch.Tensor
            position y-coordinate.
        px : Optional[torch.Tensor], optional
            Generalized momenta x-coordinate. Defaults to zero if None. The default is None.
        py : Optional[torch.Tensor], optional
            Generalized momenta x-coordinate. Defaults to zero if None. The default is None.
        mass : float, optional
            Mass of particles. The default is 1.0.
        charge : float, optional
            Charge of particles. The default is 1.0.
        integrator : str, optional
            Integrator name for system. The default is 'sympleticverlet'.
        coupling : float, optional
            Coupling constant for system. The default is 1.0.
        darwin_coupling : Optional[float], optional
            Darwin lagrangian coupling constant for system. If None, Darwin term is not considered.
            The default is None.
        """
        self.points = points.MovingPoints(x, y, px, py, mass, charge,
                                          lx, ly, cx, cy)
        self.objects = []
        self.integrator = integrators.get_integrator(integrator)
        self.coupling = coupling
        self.darwin_coupling = darwin_coupling
        
    def add_field_object(self, field_obj: fields.FieldObject):
        """
        

        Parameters
        ----------
        field_obj : fields.FieldObject
            FieldObject creator of external field.
        """
        assert isinstance(field_obj, fields.FieldObject)
        self.objects.append(field_obj)
    
    def step(self, dt: float):
        """
        

        Parameters
        ----------
        dt : float
            Step size.

        """
        self.integrator(dt, self.points, self.objects,
                        self.coupling, self.darwin_coupling)
        if self.points.periodic:
            self.points.wrap_around()

    def set_integrator(self, integrator: str):
        """
        

        Parameters
        ----------
        integrator : str
            integrator.

        """
        self.integrator = integrators.get_integrator(integrator)
        if integrator[:3] == "tao":
            self.points.make_dummy_parameters()
            
    @property
    def periodic(self):
        return self.points.periodic