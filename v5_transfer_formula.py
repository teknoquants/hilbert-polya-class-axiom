"""
Proposition 4.3 (Gelfand-Yaglom transfer formula) on a Dirichlet truncation H_L:

   u_kappa(L)/u_0(L) = det(I + kappa^2 H_L^{-1}) = prod_n (1 + kappa^2/E_n(L))
   d/dkappa log u_kappa(L) = 2 kappa Tr(H_L + kappa^2)^{-1} = 2 kappa sum_n 1/(E_n(L)+kappa^2)

Left sides: initial-value transfer (cosh/sinh propagation + barrier jumps).
Right sides: the first M eigenvalues of H_L (Pruefer + bisection) plus the
tail sum_{n>M} with E_n ~ (n pi/L)^2.
"""
import _path  # noqa
import numpy as np
from hp_lattice import eigenvalue

x0, g0, N = 1.0, 1.0, 40
L = x0 * np.log(N)
k = np.arange(2, N + 1)
xk = x0 * np.log(k)
gk = g0 * k ** -0.5


def u_end(kap):
    u, up, x = 0.0, 1.0, 0.0
    for xb, g in zip(xk, gk):
        d = xb - x
        if kap == 0.0:
            u, up = u + up * d, up
        else:
            c, s = np.cosh(kap * d), np.sinh(kap * d)
            u, up = u * c + up * s / kap, kap * u * s + up * c
        if xb < L - 1e-15:          # barrier at x = L contributes nothing (Dirichlet end)
            up += g * u
        x = xb
    return u


M = 20000
print(f"computing {M} Dirichlet eigenvalues of H_L (L = log {N}) ...")
E = np.empty(M)
hi = 1.0
for n in range(1, M + 1):
    E[n - 1] = eigenvalue(n, x0, g0, "log", N, Ehi=hi)
    hi = E[n - 1] * 1.01 + 1
ok = True
for kap in [0.5, 1.0, 2.0, 3.0]:
    tail_log = kap ** 2 * L ** 2 / (np.pi ** 2 * M)        # sum_{n>M} kappa^2/E_n
    rhs_ratio = np.exp(np.sum(np.log1p(kap ** 2 / E)) + tail_log)
    lhs_ratio = u_end(kap) / u_end(0.0)
    h = 1e-5
    lhs_d = (np.log(u_end(kap + h)) - np.log(u_end(kap - h))) / (2 * h)
    rhs_d = 2 * kap * (np.sum(1 / (E + kap ** 2)) + L ** 2 / (np.pi ** 2 * M))
    e1 = abs(lhs_ratio - rhs_ratio) / rhs_ratio
    e2 = abs(lhs_d - rhs_d) / rhs_d
    print(f"kappa={kap}: ratio {lhs_ratio:.10f} vs det {rhs_ratio:.10f} (rel {e1:.1e}); "
          f"dlog {lhs_d:.10f} vs 2k Tr {rhs_d:.10f} (rel {e2:.1e})")
    ok &= e1 < 1e-5 and e2 < 1e-5
print("PASS" if ok else "FAIL")
