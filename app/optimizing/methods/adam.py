import numpy as np
from optimizing.interface import get_subgradient
from optimizing.dtw_mean import frechet

"""
Based on:

Adam: A Method for Stochastic Optimization
https://arxiv.org/abs/1412.6980

Require: α: Stepsize
Require: β1, β2 ∈ [0, 1): Exponential decay rates for the moment estimates
Require: f(θ): Stochastic objective function with parameters θ
Require: θ0: Initial parameter vector
m0 ← 0 (Initialize 1st moment vector)
v0 ← 0 (Initialize 2nd moment vector)
t ← 0 (Initialize timestep)
while θt not converged do
    t ← t + 1
    gt ← ∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
    mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
    vt ← β2 · vt−1 + (1 − β2) · g2t (Update biased second raw moment estimate)
    mbt ← mt/(1 − βt1) (Compute bias-corrected first moment estimate)
    vbt ← vt/(1 − βt2) (Compute bias-corrected second raw moment estimate)
    θt ← θt−1 − α · mb t/(√vbt + eps) (Update parameters)
end while
return θt (Resulting parameters)
"""

def run(X, z, f, batch_size, n_coverage, n_epochs, d_converged, rng):
    N = X.shape[0]  # number of data point
    d = z.shape     # dimensions of data

    # number of update steps
    n_steps = int(np.ceil(n_coverage / batch_size))

    # initialize adam parameters 
    # (as it was done in the paper)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8

    # initialize momentum estimates
    m = np.zeros((n_steps + 1,) + d)
    v = np.zeros((n_steps + 1,) + d)

    # counter for visited samples and update steps
    n_visited_samples = 0
    t = 0
    
    for k in range(n_epochs):
        # shuffle data indices for new epoch
        perm = rng.permutation(N)

        for i in range(0, N, batch_size):
                        
            # break if number of samples to visit is reached or exceeded
            if not n_visited_samples < n_coverage:
                break
            
            # break if there is not an entire batch left 
            # (relevant for batch_size > 1)
            if N - i < batch_size:
                break

            # update step index (+1 indexed, because of m, v initialization)
            t += 1

            # get subgradients for current steps
            g = get_subgradient(X, z, i, batch_size, perm)

            # compute biased momentum estimates
            m[t] = beta1 * m[t - 1] + (1 - beta1) * g
            v[t] = beta2 * v[t - 1] + (1 - beta2) * np.square(g)

            # compute bias-corrected momentum estimates
            m_ = m[t] / (1 - np.power(beta1, t))
            v_ = v[t] / (1 - np.power(beta2, t))
            
            # actual update step
            z = z - alpha * m_ / (np.sqrt(v_) + eps_stable)

            # update number of visited samples
            n_visited_samples += batch_size
        
        # f[0] is initial value, therefore +1 indexed
        f[k + 1] = frechet(z, X)

        # check if current z is best
        if np.amin(f) == f[k + 1]:
            z_ = z

        # stop if converged
        f_diff = abs((f[k + 1] -  f[k]) / f[k])
        if f_diff < d_converged:
            break

    return z_, f

        