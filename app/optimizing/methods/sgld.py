import numpy as np
from scipy.optimize import fmin
from optimizing.interface import get_subgradient
from optimizing.dtw_mean import frechet, dtw
from math import sqrt
import logging

logger = logging.getLogger(__name__)

'''
This file implements the SSGLD algorithm - an subgradient and dtw able version
of the SGLD algorithm introduced in the following paper:
https://www.researchgate.net/publication/313857927_A_Hitting_Time_Analysis_of_Stochastic_Gradient_Langevin_Dynamics
'''
def run(X, z, f, batch_size, n_coverage, n_epochs, d_converged, rng):

    X = np.array(X) # sample input
    N = X.shape[0]  # amount of samples
    d = X.shape[1]  # dimension of each sample
    m = X.shape[2]  # dimension of each datapoint of each sample

    n_steps = int(np.ceil(n_coverage/batch_size))  # iterations to go through

    n_visited_samples = 0 # count how many samples we visited
    t = 0 # iteration step index

    kmin = np.amin(X, axis=0)  # lower limit of solution
    kmax = np.amax(X, axis=0)  # higher limit of solution

    xi = 8500 / np.abs(np.min(kmax)) # downscaling bias of gaussian noise
    lr = 0.05 # learning rate
    D = 8*sqrt((2*lr*d)/(xi/100))  # maximum distance to next update

    # initialize solution array with first value being random in solution range
    x = np.zeros((n_steps+1, d, m))
    x[0] = np.array([rng.uniform(low=kmin[i], high=kmax[i]) for i in range(d)])


    for k in range(n_epochs):
        # shuffle data indices for new epoch
        perm = rng.permutation(N)

        for i in range(0, N, batch_size):

            # if we visited enough samples already
            if not n_visited_samples < n_coverage:
                break

            # break if there is not an entire batch left
            # (relevant for batch_size > 1)
            if N - i < batch_size:
                break

            # update step index (+1 indexed, because of m, v initialization)
            t += 1

            # calculate gaussian noise
            w = np.array([[rng.normal(0, 1)] for i in range(d)])

            # calculate subgradient for computation of update candidate
            g = get_subgradient(X, x[t-1], i, batch_size, perm)

            # calculate update candidate
            y = x[t-1] - lr * g + sqrt((2*lr)/xi) * w

            # update step
            x[t] = y if np.less_equal(y, kmax).all() and np.greater_equal(y, kmin).all() and dtw(x[t-1],y) < D else x[t-1]

            n_visited_samples += batch_size

    # make possible solutions a set
    x = np.unique(x, axis=0)

    logger.info(f"Now calculating {x.shape[0]} Fréchet variations")

    # calculate the Fréchet function for every possible solution
    f_array = np.array([frechet(val, X) for val in x])
    
    # find indices of lowest fréchet function (also indices of correlating x)
    best_x = np.argmin(f_array)

    # flatten x[best_x] if it is a single element list
    result = x[best_x].flatten() if m == 1 else x[best_x]

    # return x with lowest Fréchet function value and the array of all Fréchet function values
    return result, f_array
