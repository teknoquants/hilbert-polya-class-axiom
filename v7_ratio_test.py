"""
Section 5.3 (decisive test, quick version): at the beta where E2/E1 matches the
zeta value 2.2119, E3/E1 and E10/E1 miss by more than 10%, for both classes.
Also checks that the minimum over the beta grid of the max relative error exceeds 10%.
Full tables: python generate_tables.py
"""
import _path  # noqa
import numpy as np
from hp_lattice import zeta_ratio_targets, eigenvalues

tgt = zeta_ratio_targets(10)
ok = True
for cls, grid in (("log", [1, 0.5, 0.3, 0.2, 0.115]), ("lambda", [1, 0.5, 0.3, 0.2, 0.12])):
    best = 1e9
    for b in grid:
        r = eigenvalues(1.0, b, cls, 10, 20)
        r = r / r[0]
        err = np.max(np.abs(r[1:] - tgt[1:]) / tgt[1:])
        best = min(best, err)
        print(f"{cls:6s} beta={b:5}: E2/E1={r[1]:.3f} E3/E1={r[2]:.3f} E10/E1={r[9]:.3f}  max err {100*err:.0f}%")
    # bracket the E2/E1 crossing and evaluate E3/E1 there
    lo, hi = 0.08, 0.2
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        r = eigenvalues(1.0, mid, cls, 3, 20)
        if r[1] / r[0] > tgt[1]:
            lo = mid
        else:
            hi = mid
    r = eigenvalues(1.0, 0.5 * (lo + hi), cls, 10, 20); r = r / r[0]
    e3 = (r[2] - tgt[2]) / tgt[2]; e10 = (r[9] - tgt[9]) / tgt[9]
    print(f"{cls:6s} crossing beta={0.5*(lo+hi):.4f}: E2/E1={r[1]:.4f} E3/E1={r[2]:.3f} ({100*e3:+.0f}%) "
          f"E10/E1={r[9]:.2f} ({100*e10:+.0f}%)   best max-error on grid {100*best:.0f}%")
    ok &= e3 > 0.10 and e10 > 0.10 and best > 0.10
print("PASS (Axiom (S_V) fails for both classes on the tested range)" if ok else "FAIL")
