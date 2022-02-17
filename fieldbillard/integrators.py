# -*- coding: utf-8 -*-
import torch


class NonValidIntegratorError(Exception):
    pass


def hamiltonian_rhs(system, objects, coupling, magnetic_coupling=None):
    system.zero_grad()
    hamiltonian = system.hamiltonian(objects, coupling, magnetic_coupling)
    hamiltonian.backward()
    x_rhs = system.px.grad
    y_rhs = system.py.grad
    px_rhs = -system.x.grad
    py_rhs = -system.y.grad
    return x_rhs, y_rhs, px_rhs, py_rhs


def force_rhs(system, objects, coupling):
    system.zero_grad()
    potential_energy = system.potential_energy(objects, coupling)
    potential_energy.backward()
    x_rhs = system.px.detach()/system.mass
    y_rhs = system.py.detach()/system.mass
    px_rhs = -system.x.grad
    py_rhs = -system.y.grad
    return x_rhs, y_rhs, px_rhs, py_rhs    


def euler_step(dt, system, objects=None, coupling=1.0, magnetic_coupling=None):
    dx, dy, dpx, dpy = hamiltonian_rhs(system, objects, coupling, magnetic_coupling)
    with torch.no_grad():
        system.x += dx*dt
        system.y += dy*dt
        system.px += dpx*dt
        system.py += dpy*dt


def midpoint_step(dt, system, objects=None, coupling=1.0, magnetic_coupling=None):
    oldx = system.x.detach()
    oldy = system.y.detach()
    oldpx = system.px.detach()
    oldpy = system.py.detach()
    dx, dy, dpx, dpy = hamiltonian_rhs(system, objects, coupling, magnetic_coupling)
    with torch.no_grad():
        system.x += 0.5*dx*dt
        system.y += 0.5*dy*dt
        system.px += 0.5*dpx*dt
        system.py += 0.5*dpy*dt
    dx, dy, dpx, dpy = hamiltonian_rhs(system, objects, coupling, magnetic_coupling)
    with torch.no_grad():
        system.x.copy_(oldx + dx*dt)
        system.y.copy_(oldy + dy*dt)
        system.px.copy_(oldpx + dpx*dt)
        system.py.copy_(oldpy + dpy*dt)


def runge_kutta_step(dt, system, objects=None, coupling=1.0, magnetic_coupling=None):
    def get_grad_and_update(scale):
        dx, dy, dpx, dpy = hamiltonian_rhs(system, objects, coupling, magnetic_coupling)
        if scale is not None:
            with torch.no_grad():
                system.x.copy_(oldx + scale*dx*dt)
                system.y.copy_(oldy + scale*dy*dt)
                system.px.copy_(oldpx + scale*dpx*dt)
                system.py.copy_(oldpy + scale*dpy*dt)
        return dx, dy, dpx, dpy
    oldx = system.x.detach()
    oldy = system.y.detach()
    oldpx = system.px.detach()
    oldpy = system.py.detach()
    dx1, dy1, dpx1, dpy1 = get_grad_and_update(0.5)
    dx2, dy2, dpx2, dpy2 = get_grad_and_update(0.5)
    dx3, dy3, dpx3, dpy3 = get_grad_and_update(1.0)
    dx4, dy4, dpx4, dpy4 = get_grad_and_update(None)
    with torch.no_grad():
        system.x.copy_(oldx + 1/6*(dx1 + 2*dx2 + 2*dx3 + dx4)*dt)
        system.y.copy_(oldy + 1/6*(dy1 + 2*dy2 + 2*dy3 + dy4)*dt)
        system.px.copy_(oldpx + 1/6*(dpx1 + 2*dpx2 + 2*dpx3 + dpx4)*dt)
        system.py.copy_(oldpy + 1/6*(dpy1 + 2*dpy2 + 2*dpy3 + dpy4)*dt)

def leapfrog_step(dt, system, objects=None, coupling=1.0, magnetic_coupling=None):
    #DEPRECATED
    return sympletic_euler_step(dt, system, objects=None, coupling=1.0, magnetic_coupling=None)


def sympletic_euler_step(dt, system, objects=None, coupling=1.0, magnetic_coupling=None):
    if magnetic_coupling is not None:
        raise NonValidIntegratorError("Method only valid without magnetostatics")
    _, _, dpx, dpy = force_rhs(system, objects, coupling)
    with torch.no_grad():
        system.px += dpx*dt
        system.py += dpy*dt
        system.x += system.px*dt/system.mass
        system.y += system.py*dt/system.mass


def sympletic_verlet_step(dt, system, objects=None, coupling=1.0, magnetic_coupling=None):
    if magnetic_coupling is not None:
        raise NonValidIntegratorError("Method only valid without magnetostatics")
    _, _, dpx, dpy = force_rhs(system, objects, coupling)
    with torch.no_grad():
        system.px += 0.5*dpx*dt
        system.py += 0.5*dpy*dt
        system.x += system.px*dt/system.mass
        system.y += system.py*dt/system.mass
    _, _, dpx, dpy = force_rhs(system, objects, coupling)
    with torch.no_grad():
        system.px += 0.5*dpx*dt
        system.py += 0.5*dpy*dt


def get_integrator(name):
    if name == "leapfrog":
        return leapfrog_step
    elif name == "sympleticverlet":
        return sympletic_verlet_step
    elif name == "sympleticeuler":
        return sympletic_euler_step
    elif name == 'euler':
        return euler_step
    elif name == "midpoint":
        return midpoint_step
    elif name == "rungekutta":
        return runge_kutta_step
    else:
        raise ValueError("Integrator not available")