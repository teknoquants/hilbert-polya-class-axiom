"""
Propositions 4.1-4.2: positivity, simplicity, counting bound and trace class.

  * N(E) <= 1 + x_K sqrt(E)/pi  with K the least integer such that
    A_K = 2 x0/(g0 sqrt K) + 2 x0^2/K^2 < 1/E
  * eigenvalues strictly positive and strictly increasing (simple spectrum);
    the Sturm count jumps by exactly one at each eigenvalue
  * partial sums of sum 1/E_n are printed (informational; trace class follows from the N(E) bound)
"""
import _path  # noqa
import numpy as np
from hp_lattice import count_below, eigenvalues

ok = True
for x0, g0 in [(1.0, 1.0), (1.0, 0.3), (2.0, 0.5), (0.5, 3.0)]:
    for E in [5.0, 20.0, 80.0]:
        K = 2
        while 2 * x0 / (g0 * np.sqrt(K)) + 2 * x0 ** 2 / K ** 2 >= 1.0 / E:
            K = int(K * 1.05) + 1
        bound = 1 + x0 * np.log(K) * np.sqrt(E) / np.pi
        N = int(min(3e8, max(1000, 60 * (E * x0 / g0) ** 2)))
        n = count_below(E, x0, g0, "log", N)
        print(f"x0={x0} g0={g0} E={E:5.1f}: N(E)={n:4d}  bound={bound:8.1f}")
        ok &= n <= bound
    ev = eigenvalues(x0, g0, "log", nev=30, fac=20)
    simple = np.all(np.diff(ev) > 1e-9 * ev[1:]) and ev[0] > 0
    # count jumps by exactly one across each eigenvalue
    N = int(min(3e8, max(1000, 60 * (ev[9] * 1.1 * x0 / g0) ** 2)))
    jumps = [count_below(e * (1 + 1e-4), x0, g0, "log", N) - count_below(e * (1 - 1e-4), x0, g0, "log", N)
             for e in ev[:10]]
    s15, s30 = np.sum(1 / ev[:15]), np.sum(1 / ev)
    print(f"   E1..E3={np.round(ev[:3],4)} positive&simple={simple} jumps={jumps} "
          f"sum 1/E_n: n<=15 {s15:.5f}, n<=30 {s30:.5f}")
    ok &= simple and all(j == 1 for j in jumps)
print("PASS" if ok else "FAIL")
