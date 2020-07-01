# Stoachstic Subgradient (SSG) Method for Averaging Time Series
# under Dynamic Time Warping (DTW).
#
# Translation by Khaled Sakallah, based on the Matlab code
# of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
# Original Author: David Schultz, DAI-Lab, TU Berlin, Germany, 2017
####################################################################

import sys
import importlib
import logging
import time
import numpy as np
from tqdm import tqdm
from .dtw_mean import *

logger = logging.getLogger(__name__)

def optimize_timed(*args, **kwargs):
    time1 = time.time()
    ret = optimize(*args, **kwargs)
    time2 = time.time()
    t = time2 - time1 
    return t, ret

def optimize(X, method, n_coverage=None, batch_size=1, d_converged=0.0001, init_sequence=None, return_z=False):

    # X is a 3-dim matrix consisting of possibly multivariate time series.
    #   dim 1 runs over the sample time series
    #   dim 2 runs over the length of a time series
    #   dim 3 runs over the dimension of the datapoints of a time series
    
    # import optimization method module according to parameter
    try:
        optimizing_method = importlib.import_module('.methods.' + method, package='optimizing')
    except Exception as e:
        logger.exception('Could not load optimization method: ' + str(e))
    
    N = X.shape[0]  # number of samples

    # the number of samples to visit during optimization (if not converged early)
    if n_coverage is None:
        n_coverage = 1000
    
    n_epochs = int(np.ceil(n_coverage / (N - (N % batch_size))))
    
    # initialize mean z
    if init_sequence is None:
        z = X[np.random.randint(N)]

    elif init_sequence > 0:
        z = X[int(init_sequence)]

    elif init_sequence <= 0:
        z = medoid_sequence(X)

    # vector to store frechet variations
    f = np.full(n_epochs + 1, np.nan)
    f[0] = frechet(z, X)

    # progress bar
    # batch_overspill = 1 if n_coverage % batch_size > 0 else 0
    # with tqdm(total=n_coverage - n_coverage % batch_size + batch_overspill * batch_size, position=pbar_position, file=sys.stdout) as pbar:
        
    # here the actual optimizing method is called
    # run(X, z, f, batch_size, n_coverage, n_epochs, d_converged, progress_bar)
    z, f = optimizing_method.run(X, z, f, batch_size, n_coverage, n_epochs, d_converged)

    # check if there are Nan values left in f
    last_epoch_idx = n_epochs - 1
    nan_epochs = np.where(np.isnan(f))[0]
    if len(nan_epochs) > 0:
        last_epoch_idx = nan_epochs[0] - 1
        logger.info(f"Stopped early because of convergence: [ {last_epoch_idx} / {n_epochs} ] epochs computed")

    # epoch with minimum frechet variation is best result
    f_min = np.min(f)

    if return_z:
        return f_min, z

    else:
        return f_min

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
