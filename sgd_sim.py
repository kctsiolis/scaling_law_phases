import numpy as np
import jax.numpy as jnp
from jax import random, jit
import matplotlib.pyplot as plt
import re
import argparse
import os

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

def train(v,D_vec,b,theta,gamma,B,r,W,cpts,key):
    num_cpts = np.shape(cpts)[0]
    risks = np.zeros(num_cpts)
    cpt_counter = 0
    for i in range(r):
        key, subkey = random.split(key)
        X, y = generate_data(B,v,D_vec,b,subkey)
        theta = sgd_update(gamma,theta,W,X,y)
        if (i+1) >= cpts[cpt_counter]:
            risks[cpt_counter] = risk(theta,W,D_vec,b)
            cpt_counter += 1
            print("Checkpoint {}".format(cpt_counter))

    return risks

def run_experiment(alpha,beta,v,d,B,gamma,Cmin,Cmax,mesh_size,tau,n_sims):
    D_vec = jnp.power(jnp.arange(v)+1,-2*alpha)
    b = jnp.power(jnp.arange(v)+1,-beta)

    key = random.key(0)

    flops = jnp.logspace(Cmin,Cmax,mesh_size)
    n_flops = jnp.shape(flops)[0]
    risks = np.zeros((n_sims,n_flops))

    print("Starting experiment")
    for i in range(n_sims):
        print("Simulation {}...".format(i+1))
        key, subkey_Z = random.split(key)
        one = jnp.ones(shape=(d,))
        Z = random.normal(subkey_Z, shape=(v,d)) / jnp.sqrt(d)
        W = tau / jnp.sqrt(d) * jnp.outer(b,one) + Z    
        cpts = flops // (B*d)
        r = int(cpts[-1])
        key, subkey_data = random.split(key)
        theta = jnp.zeros(d)
        risks[i,:] = train(v,D_vec,b,theta,gamma,B,r,W,cpts,subkey_data)

    risks_mean = np.mean(risks,axis=0)
    risks_std = np.std(risks,axis=0) / np.sqrt(n_sims)
    return risks_mean, risks_std

def parse_results(file):
    Cmin = int(re.search(r'\d+',re.search(r'Cmin=\d+',file).group(0)).group(0))
    Cmax = int(re.search(r'\d+',re.search(r'Cmax=\d+',file).group(0)).group(0))
    mesh_size = int(re.search(r'\d+',re.search(r'mesh_size=\d+',file).group(0)).group(0))
    d = int(re.search(r'\d+',re.search(r'd=\d+',file).group(0)).group(0))
    risks = np.load(file)

    return d, risks, Cmin, Cmax, mesh_size

def plot_results(dims,risks,Cmin,Cmax,mesh_size):
    flops = jnp.logspace(Cmin,Cmax,mesh_size)
    for j,d in enumerate(dims):
        plt.plot(flops,risks[:,j],label="d = {}".format(d))
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("flops")
    plt.ylabel("Risk")

def get_args(parser):
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Data complexity.')
    parser.add_argument('--beta', type=float, default=0.25,
                        help='Target complexity.')
    parser.add_argument('--tau', type=float, default=0,
                        help='Rank-one perturbation coefficient')
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
    parser.add_argument('--Cmax', type=int, default=9,
                        help='Logarithm (base 10) of largest number of flops to try.')
    parser.add_argument('--mesh_size', type=int, default=50,
                        help='Number of points in flops mesh.')
    parser.add_argument('--num_sims', type=int, default=4,
                        help='Number of SGD simulations to run.')
    
    args = parser.parse_args()
    
    return args
    
def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    risks_mean, risks_std = run_experiment(args.alpha,args.beta,args.v,args.d,args.B,args.gamma,
                   args.Cmin,args.Cmax,args.mesh_size,args.tau,args.num_sims)
    np.save(os.path.expanduser("~/scaling_law_phases/results/risks_mean_alpha={},beta={},tau={},v={},d={},gamma={},B={},Cmin={},Cmax={},mesh_size={},num_sims={}".format(args.alpha,args.beta,args.tau,args.v,args.d,args.gamma,args.B,args.Cmin,args.Cmax,args.mesh_size,args.num_sims)), risks_mean)
    np.save(os.path.expanduser("~/scaling_law_phases/results/risks_std_alpha={},beta={},tau={},v={},d={},gamma={},B={},Cmin={},Cmax={},mesh_size={},num_sims={}".format(args.alpha,args.beta,args.tau,args.v,args.d,args.gamma,args.B,args.Cmin,args.Cmax,args.mesh_size,args.num_sims)), risks_std)
    
if __name__ == '__main__':
    main()