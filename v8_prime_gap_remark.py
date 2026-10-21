"""
Section 4.4 remark: the confinement argument for V_Lambda needs
  (q' - q) / (sqrt(q) log q) -> 0  over consecutive prime powers q < q'.
Prints the running maximum of this ratio up to 10^7 (numerical illustration only).
"""
import _path  # noqa
import numpy as np
from hp_lattice import mangoldt

L = mangoldt(10 ** 7)
q = np.nonzero(L)[0].astype(float)
r = np.diff(q) / (np.sqrt(q[:-1]) * np.log(q[:-1]))
for lim in (1e3, 1e4, 1e5, 1e6, 1e7):
    m = q[:-1] >= lim / 10
    sel = m & (q[:-1] < lim)
    print(f"q in [{lim/10:.0e}, {lim:.0e}): max ratio = {r[sel].max():.4f}")
print("INFO (no pass/fail: illustration; admissibility of V_Lambda is open)")
