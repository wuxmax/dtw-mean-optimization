import numpy
import dtw_mean
from scipy.optimize import fmin

def f_gradient(x):
    return x # todo

def projection(x):
    return fmin(np.linalg.norm(y - x) ^ 2) # todo

def set_weights(T):
    weights = [0]*T
    for i in range(0, T):
        if i <= 2:
            weights[i] = 1
        else:
            weights[i] = 1/4 * (i + 1)
    return weights

def acceleGrad(iterations=10, x_0, D, weights=None, G):

    if weights == None:
        weights = set_weights(iterations)

    x = y = z = theta = np.array([])
    x[0] = y[0] = z[0] = x_0

    for t in range(0, iterations):
        theta[t] = 1 / weights[t]

        x[t + 1] = theta[t] * z[t] + (1 - theta[t]) * y[t]
        g[t + 1] = f_gradient(x[t + 1])
        learning_rates[i] = (2 * D) / ((G ^ 2 + sum((weights[i] ^ 2) * np.linalg.norm(g[i]))) ^ 1/2)
        z[t + 1] = projection(z[t] - weights[i] * learning_rates[t] * g[t])
        y[t + 1] = x[t + 1] - learning_rates[i] * g[t]

    return np.dot(weights, y) # todo
