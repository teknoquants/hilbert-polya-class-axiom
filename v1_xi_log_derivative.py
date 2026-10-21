"""
Section 2: the right side of Axiom (S_V) equals xi'(s)/xi(s) for s > 1:

  xi'/xi = 1/s + 1/(s-1) - (1/2) log pi + (1/2) psi(s/2) - sum_{m>=2} Lambda(m) m^{-s}

Checked against high-precision numerical differentiation of
xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s), and xi(1/2) != 0.
"""
import _path  # noqa
import mpmath as mp
mp.mp.dps = 40


def xi(s):
    return mp.mpf(1) / 2 * s * (s - 1) * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def rhs(s):
    # -zeta'/zeta = sum Lambda(m) m^-s ; use mpmath zeta derivative (exact identity)
    return 1 / s + 1 / (s - 1) - mp.log(mp.pi) / 2 + mp.digamma(s / 2) / 2 + mp.zeta(s, derivative=1) / mp.zeta(s)


def rhs_prime_sum(s, M=200000):
    from hp_lattice import mangoldt
    L = mangoldt(M)
    ps = mp.fsum(mp.mpf(L[m]) * mp.power(m, -s) for m in range(2, M + 1) if L[m])
    return 1 / s + 1 / (s - 1) - mp.log(mp.pi) / 2 + mp.digamma(s / 2) / 2 - ps


ok = True
for s in [mp.mpf("1.5"), mp.mpf(2), mp.mpf(3), mp.mpf(5)]:
    num = mp.diff(xi, s) / xi(s)
    a = rhs(s)
    e1 = abs(num - a) / abs(num)
    msg = ""
    if s >= 3:   # explicit prime sum (truncated at m <= 2000) converges fast only for larger s
        b = rhs_prime_sum(s, 2000)
        msg = f"   explicit Lambda-sum (m<=2000) rel.err={mp.nstr(abs(num - b) / abs(num),3)}"
    print(f"s={mp.nstr(s,3)}: xi'/xi numeric={mp.nstr(num,15)}  formula={mp.nstr(a,15)}  "
          f"rel.err={mp.nstr(e1,3)}" + msg)
    ok &= e1 < 1e-25
print("xi(1/2) =", mp.nstr(xi(mp.mpf(1) / 2), 15))
ok &= abs(xi(mp.mpf(1) / 2)) > 0.4
print("PASS" if ok else "FAIL")
