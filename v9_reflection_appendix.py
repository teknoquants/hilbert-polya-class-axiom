"""
Appendix A: for R = diag(P, -P), J = swap, P >= 0:
  R* = R, J* = J, J^2 = I, JRJ = -R,  <Rv,v> = 0 for v = (f,f)/sqrt2,
  <R^2 v, v> = ||P f||^2.
Finite-dimensional random test.
"""
import _path  # noqa
import numpy as np
rng = np.random.default_rng(1)
ok = True
for n in (3, 10, 50):
    d = rng.uniform(0, 0.25, n) ** 2           # (1/2 - Re rho)^2 in [0, 1/4)
    P = np.diag(d); Z = np.zeros((n, n)); I = np.eye(n)
    R = np.block([[P, Z], [Z, -P]]); J = np.block([[Z, I], [I, Z]])
    f = rng.normal(size=n) + 1j * rng.normal(size=n)
    v = np.concatenate([f, f]) / np.sqrt(2)
    checks = [np.allclose(R, R.T), np.allclose(J, J.T), np.allclose(J @ J, np.eye(2 * n)),
              np.allclose(J @ R @ J, -R), abs(np.vdot(v, R @ v)) < 1e-12,
              np.isclose(np.vdot(v, R @ R @ v).real, np.linalg.norm(P @ f) ** 2)]
    print(f"n={n}: {checks}")
    ok &= all(checks)
print("PASS" if ok else "FAIL")
