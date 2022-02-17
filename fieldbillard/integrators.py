# -*- coding: utf-8 -*-
import math
import functools

import torch


class NonValidIntegratorError(Exception):
    pass


def hamiltonian_gradients(system, objects, coupling, darwin_coupling=None,
                          dummy_q=False, dummy_p=False):
    system.zero_grad()
    hamiltonian = system.hamiltonian(objects, coupling, darwin_coupling,
                                     dummy_q, dummy_p)
    hamiltonian.backward()
    dhdxy = system.xy.grad if not dummy_q else system.xy_dummy.grad
    dhdpxy = system.pxy.grad if not dummy_p else system.pxy_dummy.grad
    return dhdxy, dhdpxy


def force_rhs(system, objects, coupling):
    system.zero_grad()
    potential_energy = system.potential_energy(objects, coupling)
    potential_energy.backward()
    xy_rhs = system.pxy.detach()/system.mass
    pxy_rhs = -system.xy.grad
    return xy_rhs, pxy_rhs


def operator_ha(system, objects, coupling, darwin_coupling, delta):
    dhdq, dhdy = hamiltonian_gradients(system, objects, coupling, darwin_coupling,
                                       False, True)
    with torch.no_grad():
        system.pxy -= delta*dhdq
        system.xy_dummy += delta*dhdy
    #return q, p - delta*dhdq, x + delta*dhdy, y


def operator_hb(system, objects, coupling, darwin_coupling, delta):
    dhdx, dhdp = hamiltonian_gradients(system, objects, coupling, darwin_coupling,
                                       True, False)
    with torch.no_grad():
        system.xy += delta*dhdp
        system.pxy_dummy -= delta*dhdx


def operator_whc(system, omega, delta):
    #Don't mind the notation in the next code
    with torch.no_grad():
        q, p, x, y = system.xy, system.pxy, system.xy_dummy, system.pxy_dummy
        qnew = 0.5*(q + x + (q - x)*math.cos(2*omega*delta) + (p - y)*math.sin(2*omega*delta))
        pnew = 0.5*(p + y + (p - y)*math.cos(2*omega*delta) - (q - x)*math.sin(2*omega*delta))
        xnew = 0.5*(q + x - (q - x)*math.cos(2*omega*delta) - (p - y)*math.sin(2*omega*delta))
        ynew = 0.5*(p + y - (p - y)*math.cos(2*omega*delta) + (q - x)*math.sin(2*omega*delta))
        system.xy.copy_(qnew)
        system.pxy.copy_(pnew)
        system.xy_dummy.copy_(xnew)
        system.pxy_dummy.copy_(ynew)
    return qnew, pnew, xnew, ynew


def sympletic_euler_step(dt, system, objects=None, coupling=1.0, darwin_coupling=None):
    if darwin_coupling is not None:
        raise NonValidIntegratorError("Method only valid without magnetostatics")
    _, dpxy = force_rhs(system, objects, coupling)
    with torch.no_grad():
        system.pxy += dpxy*dt
        system.xy += system.pxy*dt/system.mass


def sympletic_verlet_step(dt, system, objects=None, coupling=1.0, darwin_coupling=None):
    if darwin_coupling is not None:
        raise NonValidIntegratorError("Method only valid without magnetostatics")
    _, dpxy = force_rhs(system, objects, coupling)
    with torch.no_grad():
        system.pxy += 0.5*dpxy*dt
        system.xy += system.pxy*dt/system.mass
    _, dpxy = force_rhs(system, objects, coupling)
    with torch.no_grad():
        system.pxy += 0.5*dpxy*dt

def tao_step(dt, system, objects=None, coupling=1.0, darwin_coupling=None,
             omega=20.0):
    operator_ha(system, objects, coupling, darwin_coupling, dt/2)
    operator_hb(system, objects, coupling, darwin_coupling, dt/2)
    operator_whc(system, omega, dt)
    operator_hb(system, objects, coupling, darwin_coupling, dt/2)
    operator_ha(system, objects, coupling, darwin_coupling, dt/2)

def get_integrator(name):
    if name == "sympleticverlet":
        return sympletic_verlet_step
    elif name == "sympleticeuler":
        return sympletic_euler_step
    elif name == "tao20":
        return functools.partial(tao_step, omega=20.0)
    elif name == "tao80":
        return functools.partial(tao_step, omega=80.0)
    elif name == "tao320":
        return functools.partial(tao_step, omega=320.0)
    else:
        raise ValueError("Integrator not available")