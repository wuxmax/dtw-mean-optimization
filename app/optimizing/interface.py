# Stoachstic Subgradient (SSG) Method for Averaging Time Series
# under Dynamic Time Warping (DTW).
#
# Translation by Khaled Sakallah, based on the Matlab code
# of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
# Original Author: David Schultz, DAI-Lab, TU Berlin, Germany, 2017
####################################################################

import time
import numpy as np
from tqdm import tqdm
from .dtw_mean import *
import importlib
import logging
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
        for k in range(1, n_epochs + 1):
            perm = np.random.permutation(N)

            for i in range(1, N + 1, batch_size):
                subgradients = np.zeros((batch_size,) + z.shape)

                # TODO: Maybe possibility for optimization for batch_size > 1
                for j in range(batch_size):
                    x = X[perm[i + j - 1]]
                    _, p = dtw(z, x, path=True)

                    W, V = get_warp_val_mat(p)
                    
                    subgradients[j] = 2 * (V * z - W.dot(x))
                
                subgradient = np.mean(subgradients)

                update_idx = (k - 1) * N + i - 1

                # update rule: update(z, subgradient, update_idx, N)
                optimizing_method.update(z, subgradient, update_idx, N)

                f[k] = frechet(z, X)

                pbar.update(batch_size)

    f = f[0:n_epochs + 1]
    
    if return_z:
        return f[0], z

    else:
        return f, z
