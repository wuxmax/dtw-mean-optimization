import numpy
import dtw_mean
from scipy.optimize import fmin

def sgld(iterations, xi, lr, D, K):

    x = [0] * iterations + 1
    x[0] = randompick(K) #todo

    for k in range(1, iterations + 1):
        w = gauss(0, I(dxd)) #todo
        find g(x[k-1]) such that E(g(x[k-1])|x[k-1]) = gradientf(x[k-1])
        y = x[k-1] - lr * g + sqrt((2*lr)/xi) * w
        x[k] = y if y in K and dtw(x[k-1],y) < D else x[k-1]

    return f(x[kk]) where kk = argmin_k(f(x[k]))
