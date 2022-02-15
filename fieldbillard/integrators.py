# -*- coding: utf-8 -*-
import torch


def hamiltonian_rhs(system, objects, coupling):
    hamiltonian = system.hamiltonian(objects, coupling)
    hamiltonian.backward()
    x_rhs = system.px.grad
    y_rhs = system.py.grad
    px_rhs = -system.x.grad
    py_rhs = -system.x.grad
    return x_rhs, y_rhs, px_rhs, py_rhs


def force_rhs(system, objects, coupling):
    potential_energy = system.potential_energy(objects, coupling)
    potential_energy.backward()
    x_rhs = system.px.detach()/system.mass
    y_rhs = system.py.detach()/system.mass
    px_rhs = -system.x.grad
    py_rhs = -system.y.grad
    return x_rhs, y_rhs, px_rhs, py_rhs    


def euler_step(dt, system, objects=None, coupling=1.0):
    system.zero_grad()
    dx, dy, dpx, dpy = hamiltonian_rhs(system, objects, coupling)
    with torch.no_grad():
        system.x += dx*dt
        system.y += dy*dt
        system.px += dpx*dt
        system.py += dpy*dt


def leapfrog_step(dt, system, objects=None, coupling=1.0):
    system.zero_grad()
    dx, dy, dpx, dpy = force_rhs(system, objects, coupling)
    with torch.no_grad():
        system.px += dpx*dt
        system.py += dpy*dt
        system.x += system.px*dt/system.mass
        system.y += system.py*dt/system.mass
        
        
def get_integrator(name):
    if name == 'leapfrog':
        return leapfrog_step
    elif name == 'euler':
        return euler_step
    else:
        raise ValueError("Integrator not available")