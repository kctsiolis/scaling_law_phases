import numpy as np
import jax.numpy as jnp
from jax import random, jit
from functools import partial
import argparse

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

def get_batch(data,iter,B):
    idx = iter*B
    return data[0][idx:idx+B,:], data[1][idx:idx+B]

def sgd_step(iter,theta,data,B,gamma,W):
    X, y = get_batch(data,iter,B)
    return sgd_update(gamma,theta,W,X,y)

def train(D_vec,b,theta,gamma,B,r,data,W):
    body_fun = partial(sgd_step,data=data,B=B,gamma=gamma,W=W)
    for iter in range(r):
        theta = body_fun(iter,theta)
    return risk(theta,W,D_vec,b)

def run_experiment(alpha,beta,v,d,B,gamma,Cmin,Cmax,mesh_size,tau=0,n_sims=5):
    D_vec = jnp.power(jnp.arange(v)+1,-2*alpha)
    b = jnp.power(jnp.arange(v)+1,-beta)

    key = random.key(0)

    flops = jnp.logspace(Cmin,Cmax,mesh_size)
    n_flops = jnp.shape(flops)[0]
    risks = np.zeros(n_flops)

    key, subkey_Z = random.split(key)
    one = jnp.ones(shape=(d,))
    Z = random.normal(subkey_Z, shape=(v,d)) / jnp.sqrt(d)
    W = tau / jnp.sqrt(d) * jnp.outer(b,one) + Z    
    for i, C in enumerate(flops):
        print(C)
        err = 0
        r = int(C // (B*d))
        n = r*B
        for _ in range(n_sims):
            key, subkey_data = random.split(key)
            data = generate_data(n,v,D_vec,b,subkey_data)
            theta = jnp.zeros(d)
            err_k = train(D_vec,b,theta,gamma,B,r,data,W)
            err += err_k
        risks[i] = err / n_sims

    return risks

def get_args(parser):
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Data complexity.')
    parser.add_argument('--beta', type=float, default=0.25,
                        help='Target complexity.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('-B', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('-v', type=int, default=1000,
                        help='Input dimension.')
    parser.add_argument('-d', type=int,
                        help='Random feature dimension.')
    parser.add_argument('--Cmin', type=int, default=4,
                        help='Logarithm (base 10) of smallest number of flops to try.')
    parser.add_argument('--Cmax', type=int, default=10,
                        help='Logarithm (base 10) of largest number of flops to try.')
    parser.add_argument('--mesh_size', type=int, default=10,
                        help='Number of points in flops mesh.')
    
    args = parser.parse_args()
    
    return args
    
def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    risks = run_experiment(args.alpha,args.beta,args.v,args.d,args.B,args.gamma,
                   args.Cmin,args.Cmax,args.mesh_size)
    np.save("results/risks_alpha={},beta={},v={},d={},gamma={},B={},Cmin={},Cmax={},mesh_size={}".format(
        args.alpha,args.beta,args.v,args.d,args.gamma,args.B,args.Cmin,args.Cmax,args.mesh_size), risks)
    
if __name__ == '__main__':
    main()