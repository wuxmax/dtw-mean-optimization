import numpy as np

""" 
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



# run(X, z, N, batch_size, perm, epoch_idx, pbar)
def run(X, z, N, batch_size, perm, epoch_idx, pbar):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8


        