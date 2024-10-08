# -*- coding: utf-8 -*-

from scipy.integrate import odeint 
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import time

ns = 81
nc = 81
epochs = 300000

sigma = 0#1000000

beta = 0.5 
gamma = 0.25 
N = 80000000 
t0 = 0
tf = 80
y0 = [N-50, 50]
Ts = jnp.linspace(t0,tf, ns)
Tc = jnp.linspace(t0,tf, nc)
Tc = Tc.reshape(nc, 1)

width = [1] + 4*[16] + [2] 

lr = 0.001 
b1 = 0.9
b2 = 0.999
eps = 1e-08
eps_root = 0.0


@jax.jit 
def MSE(pred, true):
    return (true - pred)**2

def SIR(y, t):
    dSdt = - beta*y[0]*y[1]/N
    dIdt = beta*y[0]*y[1]/N - gamma*y[1]
    return jnp.array([dSdt, dIdt])

def GenerateDataODE(ode, y0, Ts, sigma):
    sol = odeint(ode, y0, Ts) + sigma*jax.random.normal(jax.random.PRNGKey(0), (ns, 2))
    return sol

sol = GenerateDataODE(SIR, y0, Ts, sigma)
sol = sol - (sol < 0)*sol
Ts = Ts.reshape(ns, 1)
sol = sol.reshape(ns, 2)/N
plt.plot(Ts, sol)

def SIRres(y, t, b, g):

    S = lambda t: y(t)[:, 0:1]
    I = lambda t: y(t)[:, 1:2]
    St = lambda t: jax.grad(lambda t: jnp.sum(S(t)))(t)
    It = lambda t: jax.grad(lambda t: jnp.sum(I(t)))(t)
    Sres = St(t) + b*S(t)*I(t)#/N
    Ires = It(t) - b*S(t)*I(t) + g*I(t) 
    return jnp.array([Sres**2, Ires**2]) 

initializer = jax.nn.initializers.glorot_normal()
key = jax.random.split(jax.random.PRNGKey(0),len(width)-1) 
nnparams = list()
for key,lin,lout in zip(key,width[:-1],width[1:]):
    W = initializer(key,(lin,lout),jnp.float32)
    B = initializer(key,(1,lout),jnp.float32)
    nnparams.append({'W':W,'B':B})
    
params = {'network': nnparams, 'beta': 0.3, 'gamma': 0.3}
    
@jax.jit
def forward(x,params):
    *hidden,output = params
    for layer in hidden:
        x = jax.nn.tanh(x @ layer['W'] + layer['B'])
    return x @ output['W'] + output['B']
    
@jax.jit 
def lf(params, Ts, Tc, sol):
    loss = 0
    loss = loss + jnp.mean(MSE(forward(Ts, params['network']), sol)) #1000*jnp.mean(MSE(forward(Ts, params['network']), sol))
    loss = loss + jnp.mean(MSE(SIRres(lambda t: forward(t, params['network']), Tc, params['beta'], params['gamma']),0))
    return loss

optimizer = optax.adam(lr, b1, b2, eps, eps_root)
opt_state = optimizer.init(params)
grad_loss = jax.jit(jax.grad(lf, 0))

@jax.jit
def update(opt_state, params, Ts, Tc, sol):
    grads = grad_loss(params, Ts, Tc, sol)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

for e in range(epochs):
  opt_state,params = update(opt_state,params, Ts, Tc, sol)    
  print(e)    
plt.plot(Ts, forward(Ts, params['network']), 'k')
print(params['beta'])
print(params['gamma'])