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
        
    def kinetic_energy(self):
        energy = 1/(2*self.mass)*torch.sum((self.px**2 + self.py**2))
        return energy
    
    def internal_energy(self, coupling=1.0):
        dists = torch.cdist(self.xy, self.xy) + utils.diagonal_mask(self.dim)
        energies = coupling*self.charge**2/dists
        energy = torch.sum(energies)        
        return energy

    def external_energy(self, objects=None, coupling=1.0):
        if objects is None:
            return 0.0
        energies = torch.stack([obj.potential(self.x, self.y, self.charge, coupling) for obj in objects])
        energy = torch.sum(energies)
        return energy
    
    def potential_energy(self, objects=1.0, coupling=1.0):
        return self.internal_energy(coupling) + self.external_energy(objects, coupling)
    
    def hamiltonian(self, objects=None, coupling=1.0, darwin_coupling=None):
        if isinstance(darwin_coupling, float):
            return self.darwin_hamiltonian(objects, coupling, darwin_coupling)
        else:
            ke = self.kinetic_energy() 
            pe = self.potential_energy(objects, coupling)
        return ke + pe

    def darwin_hamiltonian(self, objects=None, coupling=1.0, darwin_coupling=1.0):
        warnings.warn("This is deeply wrong and has nothing to do with darwin lagrangian")
        ie, imx, imy = self.internal_energies_and_momentum(coupling, darwin_coupling)
        #e, mx, my = ie + ee, imx + emx, imy + emy
        mx, my = imx, imy
        darwin_kinetic_energy = 1/(2*self.mass)*\
                                  torch.sum(((self.px-mx)**2 + (self.py-my)**2))
        internal_potential_energy = sum(ie)
        external_potential_energy = self.external_energy(objects, coupling)
        darwin_potential_energy = internal_potential_energy + external_potential_energy
        hamilt = darwin_kinetic_energy + darwin_potential_energy
        return hamilt
    
    def internal_energies_and_momentum(self, coupling, darwin_coupling):
        dists = torch.cdist(self.xy, self.xy) + utils.diagonal_mask(self.dim)
        energies = torch.sum(coupling*self.charge**2/dists, axis=0)
        momentum_base = darwin_coupling*self.charge**2/dists
        momentum_x = torch.sum(momentum_base*self.vx, axis=0)
        momentum_y = torch.sum(momentum_base*self.vy, axis=0)
        return energies, momentum_x, momentum_y
        
        
    @property
    def xy(self):
        return torch.stack([self.x, self.y], axis=-1)
    
    @property
    def pxy(self):
        return torch.stack([self.px, self.py], axis=-1)
    
    @property
    def vx(self): #Will not be correct in darwin case
        return self.px/self.mass
    
    @property
    def vy(self): #Will not be correct in darwin case
        return self.py/self.mass
    
    @property
    def vxy(self): #Will not be correct in darwin case
        return torch.stack([self.vx, self.vy], axis=-1)
    
    @property
    def dim(self):
        return self.x.shape[-1]
    
    def _assert_positions(self):
        assert (self.x.ndim, self.y.ndim, self.px.ndim, self.py.ndim) == ((1,)*4)
        assert (self.x.shape[-1], self.y.shape[-1], self.px.shape[-1], self.py.shape[-1]) == ((self.dim,)*4)
        
    def _register_positions_as_parameters(self, x, y, px, py):
        self.x = torch.nn.Parameter(x)
        self.y = torch.nn.Parameter(y)
        self.px = torch.nn.Parameter(px)
        self.py = torch.nn.Parameter(py)
        return