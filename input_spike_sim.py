import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import jax.numpy as jnp
from jax import random, jit
from tqdm import tqdm
import argparse
import re
import os

@jit
def loss(theta,W,data):
    X, y = data
    return jnp.mean((X @ W @ theta - y)**2)

# Sample a dataset of size n
def generate_data(n,m,D_vec,u,kappa,key):
    _, bulk_key, spike_key = random.split(key, 3)
    bulk = random.normal(bulk_key,shape=(n,m))
    spike = u * jnp.ones(shape=(m,)).T * random.normal(spike_key, shape=(n,m))
    X = D_vec * bulk + kappa**(1/2) / jnp.sqrt(m) * spike
    y = X @ u

    return X,y

@jit 
def sgd_update(gamma,theta,W,X,y):
    return theta - gamma * W.T @ X.T @ (X @ W @ theta - y)

# Approximation of risk / test error
@jit
def risk(theta,W,D_vec,u):
    Wtheta = W @ theta
    return jnp.linalg.norm(jnp.sqrt(D_vec) * (Wtheta - u))**2

def train(kappa,m,D_vec,u,theta,gamma,B,r,W,cpts,key):
    num_cpts = np.shape(cpts)[0]
    risks = np.zeros(num_cpts)
    cpt_counter = 0
    for i in tqdm(range(r)):
        key, data_key = random.split(key)
        X, y = generate_data(B,m,D_vec,u,kappa,data_key)
        theta = sgd_update(gamma,theta,W,X,y)
        if (i+1) >= cpts[cpt_counter]:
            risks[cpt_counter] = risk(theta,W,D_vec,u)
            cpt_counter += 1

    return risks

def run_experiment(alpha,beta,kappa,m,d,B,gamma,Cmin,Cmax,mesh_size,num_sims=1):
    D_vec = jnp.power(jnp.arange(m)+1,-2*alpha)
    u = jnp.power(jnp.arange(m)+1,-beta)

    key = random.key(0)

    flops = jnp.logspace(Cmin,Cmax,mesh_size)
    n_flops = jnp.shape(flops)[0]
    risks = np.zeros((num_sims,n_flops))

    gamma = gamma / jnp.sum(jnp.arange(1,m)**(-2.0 * alpha))

    print("Starting experiment: m = {}, d = {}".format(m,d))
    for i in range(num_sims):
        print("Simulation {}/{}...".format(i+1,num_sims))
        key, W_key = random.split(key)
        W = random.normal(W_key, shape=(m,d)) / jnp.sqrt(d) 
        cpts = flops // (6*B*d)
        r = int(cpts[-1])
        key, data_key = random.split(key)
        theta = jnp.zeros(d)
        risks[i,:] = train(kappa,m,D_vec,u,theta,gamma,B,r,W,cpts,data_key)
    
    risks_mean = np.mean(risks,axis=0)
    risks_std = np.std(risks,axis=0) / np.sqrt(num_sims)
    return risks_mean, risks_std

def get_args(parser):
    parser.add_argument('--alpha', type=float,
                        help='Data complexity.')
    parser.add_argument('--beta', type=float,
                        help='Target complexity.')
    parser.add_argument('--kappa', type=float,
                        help='Rank-one perturbation coefficient')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('-B', type=int, default=1,
                        help='Batch size.')
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

# def main():
#     parser = argparse.ArgumentParser()
#     args = get_args(parser)
#     risks_mean, risks_std = run_experiment(args.alpha,args.beta,args.v,args.d,args.B,args.gamma,
#                    args.Cmin,args.Cmax,args.mesh_size,args.tau,args.num_sims)
#     np.save(os.path.expanduser("~/scaling_law_phases/results/risks_mean_alpha={},beta={},tau={},v={},d={},gamma={},B={},Cmin={},Cmax={},mesh_size={},num_sims={}".format(args.alpha,args.beta,args.tau,args.v,args.d,args.gamma,args.B,args.Cmin,args.Cmax,args.mesh_size,args.num_sims)), risks_mean)
#     np.save(os.path.expanduser("~/scaling_law_phases/results/risks_std_alpha={},beta={},tau={},v={},d={},gamma={},B={},Cmin={},Cmax={},mesh_size={},num_sims={}".format(args.alpha,args.beta,args.tau,args.v,args.d,args.gamma,args.B,args.Cmin,args.Cmax,args.mesh_size,args.num_sims)), risks_std)

def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    
    alpha = args.alpha
    beta = args.beta
    kappa = args.kappa
    
    Cmin = args.Cmin
    Cmax = args.Cmax
    mesh_size = args.mesh_size
    num_sims = args.num_sims
    B = args.B
    gamma = args.gamma
    
    dims = [300,400,600,800,1200,1600,3200,6400,9600,12800]
    n_dims = len(dims)
    risks = np.zeros((mesh_size,n_dims))
    
    for j, d in enumerate(dims):
        risks_mean, risks_std = run_experiment(alpha,beta,kappa,2*d,d,B,gamma,Cmin,Cmax,mesh_size,num_sims)
        np.save(os.path.expanduser("results/risks_mean_alpha={},beta={},kappa={},m={},d={},gamma={},B={},Cmin={},Cmax={},mesh_size={},num_sims={}".format(
            alpha,beta,kappa,2*d,d,gamma,B,Cmin,Cmax,mesh_size,num_sims)), risks_mean)
        np.save(os.path.expanduser("results/risks_std_alpha={},beta={},kappa={},m={},d={},gamma={},B={},Cmin={},Cmax={},mesh_size={},num_sims={}".format(
            alpha,beta,kappa,2*d,d,gamma,B,Cmin,Cmax,mesh_size,num_sims)), risks_std)
    
if __name__ == '__main__':
    main()