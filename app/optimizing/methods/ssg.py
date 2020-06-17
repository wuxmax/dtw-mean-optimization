# Stoachstic Subgradient (SSG) Method for Averaging Time Series
# under Dynamic Time Warping (DTW).
#
# Translation by Khaled Sakallah, based on the Matlab code
# of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
# Original Author: David Schultz, DAI-Lab, TU Berlin, Germany, 2017
####################################################################

import numpy as np


def update(z, subgradient, update_idx, N):
    eta = np.linspace(0.1, 0.005, N)

    if update_idx <= eta.shape[0]:
        lr = eta[update_idx]
    else:
        lr = eta[-1]

    return z - lr * subgradient


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