"""
Lemma 5.1 (scaling): H(x0, g0) is unitarily equivalent to lambda^{-2} H(x0/lambda, lambda g0).
Hence E_n(x0, g0) = lambda^{-2} E_n(x0/lambda, lambda g0) and E_n/E_1 depends only on beta = x0 g0.
The Pruefer recursion is itself scale-covariant (sqrt(E) d and g/sqrt(E) are invariant),
so agreement is to rounding error; this confirms that the numerics respect the lemma.
"""
import _path  # noqa
import numpy as np
from hp_lattice import eigenvalues

ok = True
for cls in ("log", "lambda"):
    for x0, g0, lam in [(1.0, 0.5, 2.0), (1.0, 0.5, 0.5), (2.0, 0.3, 4.0)]:
        a = eigenvalues(x0, g0, cls, nev=6, fac=40)
        b = eigenvalues(x0 / lam, lam * g0, cls, nev=6, fac=40) / lam ** 2
        e = np.max(np.abs(a - b) / a)
        print(f"{cls:6s} x0={x0} g0={g0} lambda={lam}: max rel diff E_n vs lambda^-2 E_n' = {e:.1e}")
        ok &= e < 1e-6
    r1 = eigenvalues(1.0, 0.6, cls, 6, 40); r2 = eigenvalues(3.0, 0.2, cls, 6, 40)
    e = np.max(np.abs(r1 / r1[0] - r2 / r2[0]))
    print(f"{cls:6s} ratios at (1, 0.6) vs (3, 0.2) [same beta=0.6]: max diff {e:.1e}")
    ok &= e < 1e-6
print("PASS" if ok else "FAIL")
