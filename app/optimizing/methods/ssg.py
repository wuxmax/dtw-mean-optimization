# Stoachstic Subgradient (SSG) Method for Averaging Time Series
# under Dynamic Time Warping (DTW).
#
# Translation by Khaled Sakallah, based on the Matlab code
# of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
# Original Author: David Schultz, DAI-Lab, TU Berlin, Germany, 2017
####################################################################

import numpy as np
from optimizing.interface import get_subgradient


def run(X, z, N, batch_size, perm, epoch_idx, progress_bar):
    # learning rate schedule
    lr_min = 0.005
    eta = np.linspace(0.1, lr_min, N)

    for i in range(0, N, batch_size):

        # get_subgradient(X, z, data_idx, batch_size, perm)
        subgradient = get_subgradient(X, z, i, batch_size, perm)

        # pick learning rate 
        if epoch_idx == 0 and i <= eta.shape[0]:
            lr = eta[i]
        else:
            lr = lr_min

        # update rule
        z = z - lr * subgradient

        # only for updating the terminal progess bar
        progress_bar.update(batch_size)

    return z


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