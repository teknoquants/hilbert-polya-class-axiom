"""
Proposition 4.1 (confinement): for f in the form domain and K >= 2,

  int_{x_K}^inf |f|^2 <= ( 2 x0/(g0 sqrt K) + 2 x0^2/K^2 ) q[f],
  q[f] = int |f'|^2 + sum_k g_k |f(x_k)|^2,  x_k = x0 log k, g_k = g0 k^{-1/2}.

Tested on random continuous piecewise-linear f with f(0)=0, nodes at the
barriers (exact integrals), for several (x0, g0, K).
"""
import _path  # noqa
import numpy as np
rng = np.random.default_rng(7)
ok = True
worst = 0.0
for trial in range(300):
    x0 = rng.uniform(0.3, 3.0)
    g0 = 10 ** rng.uniform(-2, 1)
    Nb = int(rng.integers(50, 4000))
    k = np.arange(2, Nb + 1)
    xk = x0 * np.log(k)
    gk = g0 * k ** -0.5
    # node values: f(0)=0 at x=0, then values at barriers, random smooth-ish
    vals = np.cumsum(rng.normal(size=Nb - 1)) * rng.uniform(0.01, 1)
    vals *= np.exp(-rng.uniform(0, 0.01) * np.arange(Nb - 1))
    vals[-1] = 0.0  # compact support: f = 0 beyond last node
    xs = np.concatenate([[0.0], xk])
    fs = np.concatenate([[0.0], vals])
    h = np.diff(xs)
    df = np.diff(fs)
    grad = np.sum(df ** 2 / h)
    q = grad + np.sum(gk * vals ** 2)
    seg_L2 = h * (fs[:-1] ** 2 + fs[:-1] * fs[1:] + fs[1:] ** 2) / 3  # exact for linear pieces
    for K in (2, 5, 20, 100):
        if K >= Nb:
            continue
        i0 = K - 1  # segment starting at x_K is index K-1 in xs (xs[K-1] = x_K)
        tail = np.sum(seg_L2[i0:])
        bound = (2 * x0 / (g0 * np.sqrt(K)) + 2 * x0 ** 2 / K ** 2) * q
        worst = max(worst, tail / bound)
        ok &= tail <= bound * (1 + 1e-12)
print(f"max (tail / bound) over all trials = {worst:.4f}  (must be <= 1)")
print("PASS" if ok else "FAIL")
