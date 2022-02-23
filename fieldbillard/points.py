# -*- coding: utf-8 -*-
from typing import Optional, List
import math
import warnings
import itertools

import torch

from . import utils
from . import fields


class MovingPoints(torch.nn.Module):
    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 px: Optional[torch.Tensor] = None, py: Optional[torch.Tensor]=None,
                 mass: float = 1.0, charge: float = 1.0, 
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
        lx : Optional[float], optional
            Blablabla    
        ly : Optional[float], optional
            Blablabla    
        cx : Optional[float], optional
            Blablabla    
        cy : Optional[float], optional
            Blablabla    
        """
        super().__init__()
        x = x #(n, )
        y = y #(n, )
        px = torch.zeros_like(x) if px is None else px
        py = torch.zeros_like(y) if py is None else py
        self._register_positions_as_parameters(x, y, px, py)
        self._assert_positions()
        #Assert blablabla
        self.charge = charge #(n, )
        self.mass = mass #(n, )
        self.lx = lx
        self.ly = ly
        self.cx = cx
        self.cy = cy
        self.nper = 1
        
    def hamiltonian(self, objects: Optional[List[fields.FieldObject]] = None,
                    coupling: float = 1.0, darwin_coupling: Optional[float] = None,
                    dummy_q: bool = False, dummy_p: bool = False) -> torch.Tensor:
        """

        Parameters
        ----------
        objects : Optional[List[fields.FieldObject]], optional
            List of external objects generating fields. The default is None.
        coupling : float, optional
            Coupling constant for system. The default is 1.0.
        darwin_coupling : Optional[float], optional
            Darwin lagrangian coupling constant for system. If None, Darwin term is not considered.
            The default is None.
        dummy_q : bool, optional
            Whether to actually use xy parameter or dummy. Used in Tao integrators only.
            The default is False.
        dummy_p : bool, optional
            Whether to actually use xy parameter or dummy. Used in Tao integrators only.
            The default is False.

        Returns
        -------
        torch.Tensor
            Resulting hamiltonian.

        """
        xy = self.xy if not dummy_q else self.xy_dummy
        pxy = self.pxy if not dummy_p else self.pxy_dummy
        if isinstance(darwin_coupling, float):
            return self.darwin_hamiltonian(xy, pxy, objects, coupling, darwin_coupling)
        else:
            ke = self.kinetic_energy(pxy) 
            pe = self.potential_energy_(xy, objects, coupling)
        return ke + pe

    def potential_energy(self, objects=1.0, coupling=1.0, dummy_q=False, dummy_p=False):
        """

        Parameters
        ----------
        objects : Optional[List[fields.FieldObject]], optional
            List of external objects generating fields. The default is None.
        coupling : float, optional
            Coupling constant for system. The default is 1.0.
        dummy_q : bool, optional
            Whether to actually use xy parameter or dummy. Used in Tao integrators only.
            The default is False.
        dummy_p : bool, optional
            Whether to actually use xy parameter or dummy. Used in Tao integrators only.
            The default is False.

        Returns
        -------
        torch.Tensor
            Resulting hamiltonian.

        """
        xy = self.xy if not dummy_q else self.xy_dummy
        return self.potential_energy_(xy, objects, coupling)

    def kinetic_energy(self, pxy):
        """Calculates kinetic energy term (for separable hamiltonian)"""
        energy = 1/(2*self.mass)*torch.sum(pxy**2)
        return energy
    
    def internal_energy(self, xy, coupling=1.0):
        """Calculates particle interactions term (for separate hamiltonian)"""
        if self.periodic:
            return self.periodic_internal_energy(xy, coupling)
        else:
            dists = torch.cdist(xy, xy) + utils.diagonal_mask(self.dim)
            energies = coupling*self.charge**2/dists
            energy = torch.sum(energies)        
            return energy

    def periodic_internal_energy(self, xy, coupling=1.0):
        xiterator = list(range(-self.nper, self.nper + 1)) if self.lx is not None else [0]
        yiterator = list(range(-self.nper, self.nper + 1)) if self.ly is not None else [0]
        iterator = itertools.product(xiterator, yiterator)
        energies = [self.dislocated_internal_energy(xy, n, m, coupling) 
                    for n, m in iterator]
        energy = sum(energies)
        return energy

    def dislocated_internal_energy(self, xy, n=0, m=0, coupling=1.0):
        xy_dis = self.dislocate_xy(xy, n, m)
        dists = torch.cdist(xy, xy_dis) + utils.diagonal_mask(self.dim)
        energies = coupling*self.charge**2/dists
        energy = torch.sum(energies)        
        return energy

    def external_energy(self, xy, objects=None, coupling=1.0):
        """Calculates external field term"""
        if objects is None:
            return 0.0
        x, y = xy[..., 0], xy[..., 1]
        energy = sum([obj.potential(x, y, self.charge, coupling).sum() for obj in objects])
        # energies = torch.stack([obj.potential(x, y, self.charge, coupling) for obj in objects])
        # energy = torch.sum(energies)
        return energy
    
    def potential_energy_(self, xy, objects=1.0, coupling=1.0):
        """Calculates potential energy term (for separable hamiltonian)"""
        return self.internal_energy(xy, coupling) + self.external_energy(xy, objects, coupling)
            
    def darwin_hamiltonian(self, xy, pxy, objects=None, coupling=1.0, darwin_coupling=1.0):
        """Calculates darwin hamiltonian"""
        if self.periodic:
            raise NotImplementedError
        internal_energy = self.darwin_energies(xy, pxy, coupling, darwin_coupling)
        external_energy = self.external_energy(xy, objects, coupling)
        hamilt = internal_energy + external_energy
        return hamilt
    
    def darwin_energies(self, xy, pxy, coupling, darwin_coupling):
        """Calculate internal terms for darwin hamiltonian"""
        dists = torch.cdist(xy, xy) + utils.diagonal_mask(self.dim)
        coulomb_energy = torch.sum(coupling*self.charge**2/dists)
        darwin_base = darwin_coupling*self.charge**2/(2*dists*self.mass**2)
        inner_products = torch.sum(pxy[None, :, :]*pxy[:, None, :], dim=-1)
        projections = torch.sum((xy[:, None, :] - xy[None, :, :])*pxy, axis=-1)/(dists)
        inner_prod_term = darwin_base*torch.sum((inner_products))
        projections_term = darwin_base*torch.sum(projections*projections.T)
        darwin_energy = torch.sum(inner_prod_term + projections_term)
        kinetic_energy = self.kinetic_energy(pxy)
        rke_base = darwin_coupling/(8*self.mass**3)*darwin_coupling
        relativistic_kinetic_energy = rke_base*torch.sum(torch.diag(inner_products))
        energy = coulomb_energy + darwin_energy + kinetic_energy + relativistic_kinetic_energy
        return energy
    
    def wrap_around(self):
        with torch.no_grad():
            self.xy.copy_(
                utils.wrap_from_center(self.xy.data,
                                       [self.lx, self.ly],
                                       [self.cx, self.cy]))

    def dislocate_xy(self, xy, n, m):
        lx = self.lx if self.lx is not None else 0.0
        ly = self.ly if self.ly is not None else 0.0
        xy_dis = xy + torch.tensor([n*lx, m*ly])
        return xy_dis

    @property
    def x(self):
        return self.xy[..., 0]
    
    @property
    def y(self):
        return self.xy[..., 1]
    
    @property
    def px(self):
        return self.pxy[..., 0]
    
    @property
    def py(self):
        return self.pxy[..., 1]
    
    @property
    def dim(self):
        return self.x.shape[-1]
    
    @property
    def periodic(self):
        return self.lx is not None or self.ly is not None

    def _assert_positions(self):
        assert (self.x.ndim, self.y.ndim, self.px.ndim, self.py.ndim) == ((1,)*4)
        assert (self.x.shape[-1], self.y.shape[-1], self.px.shape[-1], self.py.shape[-1]) == ((self.dim,)*4)
        
    def _register_positions_as_parameters(self, x, y, px, py):
        xy = torch.stack([x, y], dim=-1)
        pxy = torch.stack([px, py], dim=-1)
        self.xy = torch.nn.Parameter(xy)
        self.pxy = torch.nn.Parameter(pxy)
        self.xy_dummy = None
        self.pxy_dummy = None
        return