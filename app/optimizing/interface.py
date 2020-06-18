# Stoachstic Subgradient (SSG) Method for Averaging Time Series
# under Dynamic Time Warping (DTW).
#
# Translation by Khaled Sakallah, based on the Matlab code
# of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
# Original Author: David Schultz, DAI-Lab, TU Berlin, Germany, 2017
####################################################################

import importlib
import logging
import time
import numpy as np
from tqdm import tqdm
from .dtw_mean import *

from numba import jit

logger = logging.getLogger(__name__)

def optimize_timed(*args, **kwargs):
    time1 = time.time()
    ret = optimize(*args, **kwargs)
    time2 = time.time()
    t = time2 - time1 
    return t, ret

def optimize(X, method, n_epochs=None, batch_size=1, init_sequence=None, return_z=False):
    # Inputs
    # X is a 3-dim matrix consisting of possibli multivariate time series.
    #   dim 1 runs over the sample time series
    #   dim 2 runs over the length of a time series
    #   dim 3 runs over the dimension of the datapoints of a time series
    #
    # Optional Inputs
    # n_epochs        is the number of epochs
    # eta             is a vector of step sizes, eta(i) is used in the i-th update
    # init_sequence   if None  --> use a random sample of X
    #                 if > 0   --> use X[init_sequence]
    #                 if <= 0  --> use medoid of X
    #                 if it is a time series --> use it
    # return_f        if True  --> Frechet variations for each epoch are returned
    #
    # Outputs
    # z               the solution found by SSG (an approximate sample mean under dynamic time warping)
    # f               Vector of Frechet variations. Is only returned if return_f=True
    
    # import optimization method module according to parameter
    try:
        optimizing_method = importlib.import_module('.methods.' + method, package='optimizing')
    except Exception as e:
        logger.exception('Could not load optimization method: ' + str(e))
    
    N = X.shape[0]  # number of samples

    if n_epochs is None:
        n_updates = 1000
        n_epochs = int(np.ceil(n_updates / N))

    # initialize mean z
    if init_sequence is None:
        z = X[np.random.randint(N)]

    elif init_sequence > 0:
        z = X[int(init_sequence)]

    elif init_sequence <= 0:
        z = medoid_sequence(X)
    
    f = np.zeros(n_epochs + 1)
    f[0] = frechet(z, X)

    # optimization
    with tqdm(total=n_epochs * N) as pbar:
        for k in range(n_epochs):
            
            # here the actual optimizing method is called
            # run(X, z, f, batch_size, n_epochs, progress_bar)
            z, f = optimizing_method.run(X, z, f, batch_size, n_epochs, progress_bar=pbar)

    # f = f[0:n_epochs + 1]
    
    if return_z:
        return f[-1], z

    else:
        return f[-1]

@jit(forceobj=True)
def get_subgradient(X, z, data_idx, batch_size, perm):
    subgradients = np.zeros((batch_size,) + z.shape)

    for j in range(batch_size):    
        x = X[perm[data_idx + j]] 

        _, p = dtw(z, x, path=True)
        W, V = get_warp_val_mat(p)                
        subgradients[j] = 2 * (V * z - W.dot(x))
        subgradient = 2 * (V * z - W.dot(x))

    subgradient = np.mean(subgradients, axis=0)

    # logger.info(f"Shape of z: {z.shape} | Shape of subgradient: {subgradient.shape} | Shape of subgradientS: {subgradients.shape}")
    # logger.info(f"Subgradient values:\n{subgradient}")
    
    return subgradient
