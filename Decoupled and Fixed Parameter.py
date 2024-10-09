# -*- coding: utf-8 -*-

from scipy.integrate import odeint 
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import time


@jax.jit 
def MSE_SA(pred, true, w):
    return (w * (true - pred)) ** 2

@jax.jit 
def MSE(pred, true):
    return (true - pred)**2



#Adam parameters

lr       = 0.001 
b1       = 0.9
b2       = 0.999
eps      = 1e-08
eps_root = 0.0

#Training points/ data gen & settings

ns = 81
nc = 81#81#81#100#200#180#80 
epochs = 500000#100000
sigma = 0

#ODE problem settings

t_0 = 0
tf   = 80
y_0 = [80000000-50, 50]
N   = 80000000
Ts = jnp.linspace(t_0, tf, ns)
Tc = jnp.linspace(t_0, tf, nc)

def gamma(t):
    return 0.25

def beta(t):
    return 0.5

def SIR(y, t):
    dSdt = - beta(t)*y[0]*y[1]/N
    dIdt = beta(t)*y[0]*y[1]/N - gamma(t)*y[1]
    return jnp.array([dSdt, dIdt])

def cSIRres(S, I, BETA, t): # FIXME
    Sprime = lambda t: jax.grad(lambda t: jnp.sum(S(t)))(t)[:, 0]
    Iprime = lambda t: jax.grad(lambda t: jnp.sum(I(t)))(t)[:, 0]
    Sres = Sprime(t) + BETA*S(t)*I(t)
    Ires = Iprime(t) - BETA*S(t)*I(t) + gamma(t)*I(t)
    return Sres**2 + Ires**2

def GenerateDataODE(ode, y0, Ts, sigma):
    sol = odeint(ode, y0, Ts) + sigma*jax.random.normal(jax.random.PRNGKey(0), (ns, 2))
    return sol

sol = GenerateDataODE(SIR, y_0, Ts, sigma)/N
sol = sol - (sol < 0)*sol
sol = sol.reshape(ns, 2)
Ts = Ts.reshape(ns, 1)
Tc = Tc.reshape(nc, 1)
plt.plot(Ts, sol)


params = {'inverse': 0.3}
widths = {'S': [1] + 4*[16] + [1], 'I': [1] + 4*[16] + [1]}
initializer = jax.nn.initializers.glorot_normal()
key = jax.random.split(jax.random.PRNGKey(0),len(widths)-1) #Seed for initialization

for compartment in widths:
    exec(compartment + "nn = list()")
    key = jax.random.split(jax.random.PRNGKey(0),len(widths[compartment])-1)
    for key,lin,lout in zip(key,widths[compartment][:-1],widths[compartment][1:]):
        W = initializer(key,(lin,lout),jnp.float32)
        B = initializer(key,(1,lout),jnp.float32)
        exec(compartment + "nn.append({'W':W,'B':B})")
    exec('params.update({ \'' + compartment + '\' : ' + compartment + 'nn' + ' })')

@jax.jit
def forward(x, params):
    *hidden,output = params
    for layer in hidden:
        x = jax.nn.tanh(x @ layer['W'] + layer['B'])
    return x @ output['W'] + output['B']

@jax.jit
def loss_func(params, Ts, Tc, sol, w=1):
    loss = 0
    loss += w*jnp.mean(MSE(cSIRres(lambda t: (forward(t, params['S'])[:, 0]), lambda t: (forward(t, params['I'])[:, 0]), params['inverse'], Tc), 0))
    loss += jnp.mean(MSE(forward(Ts, params['S']).reshape(ns,), sol[:, 0]))
    loss += jnp.mean(MSE(forward(Ts, params['I']).reshape(ns,), sol[:, 1]))
    return loss

optimizer = optax.adam(lr, b1, b2, eps, eps_root)
opt_state = optimizer.init(params)
grad_loss = jax.jit(jax.grad(loss_func, 0))

@jax.jit
def update(opt_state, params, Ts, Tc, sol, w):
    grads = grad_loss(params, Ts, Tc, sol)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

for e in range(epochs):
  opt_state,params = update(opt_state,params, Ts, Tc, sol, 100)    
  print(e)    
plt.plot(Ts, forward(Ts, params['S']), 'k')
plt.plot(Ts, forward(Ts, params['I']), 'k')

print((params['inverse']))
