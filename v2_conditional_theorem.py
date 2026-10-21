"""
Theorem 3.1 / Remark 3.3, consistency check.

If E_n = gamma_n^2 / c^2 (the spectrum that (S_V) forces), then
  (a) D'(s)/D(s) = 2(s-1/2) sum_n 1/(c^2 E_n + (s-1/2)^2) equals xi'(s)/xi(s) for s > 1,
  (b) xi(s) = xi(1/2) * prod_n (1 + (s-1/2)^2/(c^2 E_n))     (identity (D)),
  (c) D vanishes exactly at s = 1/2 +- i gamma_n.
These are Hadamard-product identities for xi (true unconditionally for the zeros
on the line used here); the script checks the normalisation and the formulas
used in the proof. Sums are truncated at M zeros with the standard tail
estimate from the zero density (1/2pi) log(t/2pi).
"""
import _path  # noqa
import mpmath as mp
mp.mp.dps = 30
M = 300
print(f"computing {M} zeta zeros ...")
g = [mp.zetazero(n).imag for n in range(1, M + 1)]


def xi(s):
    return mp.mpf(1) / 2 * s * (s - 1) * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def tail_logderiv(s):
    a = s - mp.mpf(1) / 2
    T = g[-1]
    # sum_{gamma>T} 2a/(gamma^2+a^2) ~ int_T^inf 2a/t^2 * (1/2pi) log(t/2pi) dt
    return a / mp.pi * (mp.log(T / (2 * mp.pi)) + 1) / T


def tail_logprod(s):
    a2 = (s - mp.mpf(1) / 2) ** 2
    T = g[-1]
    # sum_{gamma>T} log(1+a^2/gamma^2) ~ a^2 int_T^inf (1/2pi) log(t/2pi)/t^2 dt
    return a2 / (2 * mp.pi) * (mp.log(T / (2 * mp.pi)) + 1) / T


ok = True
for s in [mp.mpf(2), mp.mpf(3), mp.mpf("1.5")]:
    a = s - mp.mpf(1) / 2
    lhs = 2 * a * mp.fsum(1 / (gg ** 2 + a ** 2) for gg in g) + tail_logderiv(s)
    ref = mp.diff(xi, s) / xi(s)
    D = mp.fprod(1 + a ** 2 / gg ** 2 for gg in g) * mp.e ** tail_logprod(s)
    e1 = abs(lhs - ref) / abs(ref)
    e2 = abs(xi(mp.mpf(1) / 2) * D - xi(s)) / abs(xi(s))
    print(f"s={mp.nstr(s,3)}: D'/D={mp.nstr(lhs,12)} xi'/xi={mp.nstr(ref,12)} rel.err={mp.nstr(e1,2)} | "
          f"xi(1/2)D(s)={mp.nstr(xi(mp.mpf(1)/2)*D,12)} xi(s)={mp.nstr(xi(s),12)} rel.err={mp.nstr(e2,2)}")
    ok &= e1 < 1e-3 and e2 < 1e-3
for n in (0, 1, 2):
    s = mp.mpf(1) / 2 + 1j * g[n]
    print(f"|xi(1/2 + i gamma_{n+1})| = {mp.nstr(abs(xi(s)),3)}")
    ok &= abs(xi(s)) < 1e-20
print("PASS" if ok else "FAIL")
