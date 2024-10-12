import numpy as np
import jax.numpy as jnp
from jax import random, jit
from matplotlib import pyplot as plt

@jit
def loss(theta,W,data):
    X, y = data
    return jnp.mean((X @ W @ theta - y)**2)

# Sample a dataset of size n
def generate_data(n,v,D_vec,b,key):
    Z = random.normal(key,shape=(n,v))
    X = D_vec * Z
    y = X @ b

    return X,y

@jit 
def sgd_update(gamma,theta,W,X,y):
    return theta - gamma * W.T @ X.T @ (X @ W @ theta - y)

# Approximation of risk / test error
@jit
def risk(theta,W,D_vec,b):
    Wtheta = W @ theta
    return jnp.sum(jnp.diag(jnp.outer(Wtheta,Wtheta)) * D_vec) + jnp.sum(D_vec * b**2) - \
        2 * jnp.sum(jnp.diag(W @ jnp.outer(theta,b)) * D_vec)

def train(alpha,beta,gamma,B,v,d,r,W,key):
    D_vec = jnp.power(jnp.arange(v)+1,-2*alpha)
    b = jnp.power(jnp.arange(v)+1,-beta)

    #Initialize data
    theta = jnp.zeros(d)

    #Sample dataset
    for _ in range(r):
        key, subkey = random.split(key)
        X, y = generate_data(B,v,D_vec,b,subkey)
        theta = sgd_update(gamma,theta,W,X,y)

    return risk(theta,W,D_vec,b)

