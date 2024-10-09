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

@jax.jit
def L2error(pred,true):
    return jnp.sqrt(jnp.sum((true - pred)**2))/jnp.sqrt(jnp.sum(true ** 2))

#Adam parameters

lr       = 0.001 
b1       = 0.9
b2       = 0.999
eps      = 1e-08
eps_root = 0.0

#Training points/ data gen & settings

ns = 81
nc = 81#81#81#100#200#180#80 
epochs = 200000#200000#100000
sigma = 0

#ODE problem settings

t_0 = 0
tf   = 1
y_0 = [80000000-50, 50]
N   = 80000000
Ts = jnp.linspace(t_0, tf, ns)
Tc = jnp.linspace(t_0, tf, nc)

def gamma(t):
    return 0.25

def beta(t):
    return 0.5 if t < 60 else 0.3

def SIR(y, t):
    dSdt = - 80*beta(t)*y[0]*y[1]/N
    dIdt = 80*beta(t)*y[0]*y[1]/N - 80*gamma(t)*y[1]
    return jnp.array([dSdt, dIdt])

#def cSIRres(S, I, BETA, t): # FIXME
#    Sprime = lambda t: jax.grad(lambda t: jnp.sum(S(t)))(t)[:, 0]
#    Iprime = lambda t: jax.grad(lambda t: jnp.sum(I(t)))(t)[:, 0]
#    Sres = Sprime(t) + BETA*S(t)*I(t)
#    Ires = Iprime(t) - BETA*S(t)*I(t) + gamma(t)*I(t)
#    return Sres**2 + Ires**2

def vSIRres(S, I , BETA, t):
    Sprime = lambda t: jax.grad(lambda t: jnp.sum(S(t)))(t)[:, 0]
    Iprime = lambda t: jax.grad(lambda t: jnp.sum(I(t)))(t)[:, 0]
    Sres = Sprime(t) + 80*jax.nn.sigmoid(BETA(t))*S(t)*I(t)
    Ires = Iprime(t) - 80*jax.nn.sigmoid(BETA(t))*S(t)*I(t) + 80*gamma(t)*I(t)
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


params = {}
widths = {'S': [1] + 4*[16] + [1], 'I': [1] + 4*[16] + [1], 'beta': [1] + 4*[16] + [1]}
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
def forward(x, params, activation = jax.nn.tanh):
    *hidden,output = params
    for layer in hidden:
        x = jax.nn.tanh(x @ layer['W'] + layer['B'])
    return x @ output['W'] + output['B']

@jax.jit
def loss_func(params, Ts, Tc, sol, w=1, epsilon = 1):
    loss = 0
    residue = vSIRres(lambda t: (forward(t, params['S'])[:, 0]), lambda t: (forward(t, params['I'])[:, 0]), lambda t: (forward(t, params['beta'])[:, 0]), Tc)
    S_data_loss = MSE(forward(Ts, params['S']).reshape(ns,), sol[:, 0])
    I_data_loss = MSE(forward(Ts, params['I']).reshape(ns,), sol[:, 1])
    
    causal_res_w  = jnp.exp(jnp.array([-epsilon * jnp.sum(residue[0:k]) for k in range(nc)])) 
    causal_Sdata_w = jnp.exp(jnp.array([-epsilon * jnp.sum(S_data_loss[0:k]) for k in range(ns)])) 
    causal_Idata_w = jnp.exp(jnp.array([-epsilon * jnp.sum(I_data_loss[0:k]) for k in range(ns)])) 

    loss += w * jnp.mean(causal_res_w * residue)
    loss += jnp.mean(causal_Sdata_w * S_data_loss)
    loss += jnp.mean(causal_Idata_w * I_data_loss)
    #loss += jnp.mean(jnp.exp(- epsilon * jnp.sum(residue[])))
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

t0 = time.time()
for e in range(epochs):
  opt_state,params = update(opt_state,params, Ts, Tc, sol, 1)    
  if e % 1000 == 0:
      l2_test = (L2error(jax.nn.sigmoid(forward(Tc, params['beta'])), jnp.array(list(map(beta, Tc))))).tolist()# + L2error(forward(Ts, params['I']), sol[:, 1])).tolist()
      l = 'Epoch: ' + str(e) + ' Time: ' + str(round(time.time() - t0,2)) + ' s Loss: ' + str(jnp.round(loss_func(params, Ts, Tc, sol),6)) + ' L2 error: ' + str(jnp.round(l2_test,6))
      print(l)
plt.figure(dpi=600)
plt.plot(Ts, sol)
plt.plot(Ts, forward(Ts, params['S']), 'r', label=r'$\mathscr{NN}_S$')
plt.plot(Ts, forward(Ts, params['I']), 'b', label=r'$\mathscr{NN}_I$')
plt.plot(Ts, list(map(beta, Ts)), 'c', label=r'$\beta$')
plt.plot(Ts, jax.nn.sigmoid(forward(Ts, params['beta'])), 'k', label=r'$\mathscr{NN}_\beta$')
plt.legend()
#print((params['inverse']))
