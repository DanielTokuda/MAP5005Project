# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from scipy.integrate import odeint 
import matplotlib.pyplot as plt

#%%

class ODE:
    def __init__(self, equation, residue, parameters, ndim):
        self.equation = equation
        self.residue = residue
        self.parameters = parameters
        self.ndim = ndim
    
    def gen_data(self, t0, T, y0, ns, sigma, key, positive = True):
        Ts = jnp.linspace(t0, T, ns)
        sol = odeint(lambda y,t: self.equation(y, t, list(self.parameters.values())), y0, Ts) + sigma*jax.random.normal(jax.random.PRNGKey(key), (ns, self.ndim))
        if positive:
            sol = sol - (sol < 0)*sol
        Ts.reshape(ns, 1)
        sol.reshape(ns, self.ndim)
        self.Ts = Ts
        self.sol = sol
        return Ts, sol
    
    def plot_data(self, pop_scaled = False, styles = 'k'):
        if pop_scaled:
            plt.plot(self.Ts, self.sol/self.parameters['N'], styles)
        else:
            plt.plot(self.Ts, self.sol, styles)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$y$')
        return

#%%    
    
def cSIR(y, t, parameters):
    dSdt = - parameters[0]*y[0]*y[1]/parameters[2]
    dIdt =   parameters[0]*y[0]*y[1]/parameters[2] - parameters[1]*y[1]
    return jnp.array([dSdt, dIdt])



def cSIRres(S, I, t, parameters, mask = lambda x : x):
    dSdt = lambda t: jax.grad(S)(t)
    dIdt = lambda t: jax.grad(I)(t)
    Sres = dSdt(t) + parameters[0]*S(t)*I(t)
    Ires = dIdt(t) - parameters[0]*S(t)*I(t)- parameters[1]*I(t)
    return Sres**2 + Ires**2




#%%

def cSEIR(y, t, parameters):
    dSdt = - parameters[0]*y[0]*y[2]/parameters[3]
    dEdt =   parameters[0]*y[0]*y[2]/parameters[3] - parameters[1]*y[1]
    dIdt =   parameters[1]*y[1] - parameters[2]*y[2]
    return jnp.array([dSdt, dEdt, dIdt])
    
def cSEIRres(S, E, I, t, parameters, mask = lambda x : x):
    dSdt = lambda t: jax.grad(S)(t)
    dEdt = lambda t: jax.grad(E)(t)
    dIdt = lambda t: jax.grad(I)(t)
    Sres = dSdt(t) + parameters[0]*S(t)*I(t)
    Eres = dEdt(t) - parameters[0]*S(t)*I(t) + parameters[1]*E(t)
    Ires = dIdt(t) - parameters[1]*E(t) + parameters[2]*I(t)
    return Sres**2 + Eres**2 + Ires**2

#%%

def vSIR(y, t, parameters):
    dSdt = - parameters[0](t)*y[0]*y[1]/parameters[2]
    dIdt = parameters[0](t)*y[0]*y[1]/parameters[2] - parameters[1](t)*y[1]
    return jnp.array([dSdt, dIdt])

def vSIRres(S, I, t, parameters, mask = lambda x : x):
    dSdt = lambda t: jax.grad(S)(t)
    dIdt = lambda t: jax.grad(I)(t)
    Sres = dSdt(t) + parameters[0](t)*S(t)*I(t)
    Ires = dIdt(t) - parameters[0](t)*S(t)*I(t) - parameters[1]*I(t)
    return Sres**2 + Ires**2

#%%

def vSEIR(y, t, parameters):
    dSdt = - parameters[0](t)*y[0]*y[2]/parameters[3]
    dEdt =   parameters[0](t)*y[0]*y[2]/parameters[3] - parameters[1](t)*y[1]
    dIdt =   parameters[1](t)*y[1] - parameters[2](t)*y[2]
    return jnp.array([dSdt, dEdt, dIdt])

def vSEIRres(S, E, I, t, parameters, mask = lambda x : x):
    dSdt = lambda t: jax.grad(S)(t)
    dEdt = lambda t: jax.grad(E)(t)
    dIdt = lambda t: jax.grad(I)(t)
    Sres = dSdt(t) + parameters[0](t)*S(t)*I(t)
    Eres = dEdt(t) - parameters[0](t)*S(t)*I(t) + parameters[1](t)*E(t)
    Ires = dIdt(t) - parameters[1](t)*E(t) + parameters[2](t)*I(t)
    return Sres**2 + Eres**2 + Ires**2

#%%

