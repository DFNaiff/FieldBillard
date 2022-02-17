# -*- coding: utf-8 -*-
import math
import warnings

import torch

from . import utils


class MovingPoints(torch.nn.Module):
    def __init__(self, x, y, px=None, py=None, mass=1.0, charge=1.0):
        """
        Parameters
        ----------
        x : torch.Tensor
            position x-coordinate.
        y : torch.Tensor
            position y-coordinate.
        px : torch.Tensor or None
            Generalized momenta x-coordinate. Defaults to zero if None.
            The default is None.
        px : torch.Tensor or None
            Generalized momenta y-coordinate. Defaults to zero if None.
            The default is None.
        mass : float
            Mass of particles. The default is 1.0.
        charge : float
            Charge of particles. The default is 1.0.
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
        
    def kinetic_energy(self, pxy):
        energy = 1/(2*self.mass)*torch.sum(pxy**2)
        return energy
    
    def internal_energy(self, xy, coupling=1.0):
        dists = torch.cdist(xy, xy) + utils.diagonal_mask(self.dim)
        energies = coupling*self.charge**2/dists
        energy = torch.sum(energies)        
        return energy

    def external_energy(self, xy, objects=None, coupling=1.0):
        if objects is None:
            return 0.0
        x, y = xy[..., 0], xy[..., 1]
        energies = torch.stack([obj.potential(x, y, self.charge, coupling) for obj in objects])
        energy = torch.sum(energies)
        return energy
    
    def potential_energy_(self, xy, objects=1.0, coupling=1.0):
        return self.internal_energy(xy, coupling) + self.external_energy(xy, objects, coupling)
    
    def potential_energy(self, objects=1.0, coupling=1.0, dummy_q=False, dummy_p=False):
        xy = self.xy if not dummy_q else self.xy_dummy
        return self.potential_energy_(xy, objects, coupling)
        
    def hamiltonian(self, objects=None, coupling=1.0, darwin_coupling=None,
                    dummy_q=False, dummy_p=False):
        xy = self.xy if not dummy_q else self.xy_dummy
        pxy = self.pxy if not dummy_p else self.pxy_dummy
        if isinstance(darwin_coupling, float):
            return self.darwin_hamiltonian(xy, pxy, objects, coupling, darwin_coupling)
        else:
            ke = self.kinetic_energy(pxy) 
            pe = self.potential_energy_(xy, objects, coupling)
        return ke + pe

    def darwin_hamiltonian(self, xy, pxy, objects=None, coupling=1.0, darwin_coupling=1.0):
        internal_energy = self.darwin_energies(xy, pxy, coupling, darwin_coupling)
        external_energy = self.external_energy(xy, objects, coupling)
        hamilt = internal_energy + external_energy
        return hamilt
    
    def darwin_energies(self, xy, pxy, coupling, darwin_coupling):
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
        
    def make_dummy_parameters(self):
        self.xy_dummy = torch.nn.Parameter(self.xy.detach())
        self.pxy_dummy = torch.nn.Parameter(self.pxy.detach())
        
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

    # @property
    # def vx(self): #Will not be correct in darwin case
    #     return self.px/self.mass
    
    # @property
    # def vy(self): #Will not be correct in darwin case
    #     return self.py/self.mass
    
    # @property
    # def vxy(self): #Will not be correct in darwin case
    #     return torch.stack([self.vx, self.vy], axis=-1)
    
    @property
    def dim(self):
        return self.x.shape[-1]
    
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