# Stoachstic Subgradient (SSG) Method for Averaging Time Series
# under Dynamic Time Warping (DTW).
#
# Translation by Khaled Sakallah, based on the Matlab code
# of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
# Original Author: David Schultz, DAI-Lab, TU Berlin, Germany, 2017
####################################################################

import numpy as np
from optimizing.interface import get_subgradient
from optimizing.dtw_mean import frechet

import logging
logger = logging.getLogger(__name__)

def run(X, z, f, batch_size, n_coverage, n_epochs, d_converged, progress_bar):
    N = X.shape[0]
    
    # learning rate schedule
    n_steps = int(np.ceil(n_coverage / batch_size))
    lr_min = 0.005
    eta = np.linspace(0.05, lr_min, n_steps)

    n_visited_samples = 0

    for k in range(n_epochs):
        # shuffle data indices for new epoch
        perm = np.random.permutation(N)

        for i in range(0, N, batch_size):
            
            # break if number of samples to visit is reached or exceeded
            if not n_visited_samples < n_coverage:
                break

            # break if there is not an entire batch left 
            # (relevant for batch_size > 1)
            if N - i < batch_size:
                break
            
            # get_subgradient(X, z, data_idx, batch_size, perm)
            subgradient = get_subgradient(X, z, i, batch_size, perm)

            # pick learning rate 
            if k == 0 and i < eta.shape[0]:
                lr = eta[i]
            else:
                lr = lr_min

            # update rule
            z = z - lr * subgradient

            # only for updating the terminal progess bar
            progress_bar.update(batch_size)

            n_visited_samples += batch_size
        
        # f[0] is initial value, therefore +1 indexed
        f[k + 1] = frechet(z, X)

        # stop if converged
        f_diff = abs((f[k + 1] -  f[k]) / f[k])
        if f_diff < d_converged:
            break

    return z, f


# class SSG:
#     def __init__(self, N):
#         self.eta = np.linspace(0.1, 0.005, N)


#     def update(self, z, subgradient, update_idx, N):
#         eta = self.eta

#         if update_idx <= eta.shape[0]:
#             lr = eta[update_idx]
#         else:
#             lr = eta[-1]

#         return z - lr * subgradient