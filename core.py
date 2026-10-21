"""
Core numerics for

  "A Class Axiom for Hilbert-Polya Operators: Exact Spectral Theory of
   Logarithmic delta-Lattices and a Decisive Test"
   R. Cooperman and A.E. Petrunina V.V. (2026)

Operator (units hbar = 2m = 1), Dirichlet condition at x = 0:

    H = -d^2/dx^2 + sum_{k>=2} g_k delta(x - x0 log k)

    class "log"    : g_k = g0 * k^(-1/2)
    class "lambda" : g_k = g0 * Lambda(k) * k^(-1/2)   (von Mangoldt weights)

Eigenvalues of the Dirichlet truncation H_L, L = x0 log N, are computed with a
Pruefer angle:  u = r sin(theta), u' = sqrt(E) r cos(theta).
  * free propagation over a gap d:   theta -> theta + sqrt(E) d
  * barrier of strength g:           cot(theta) -> cot(theta) + g / sqrt(E)
                                     (branch of theta unchanged, u is continuous)
Sturm oscillation: #{eigenvalues of H_L below E} = floor(theta(L) / pi).
Each E_n is located by bisection on this count (Section 5.2 of the paper).
"""
import numpy as np
import numba

# First ten positive ordinates of nontrivial zeta zeros (LMFDB / Odlyzko tables)
GAMMA = np.array([14.134725142, 21.022039639, 25.010857580, 30.424876126,
                  32.935061588, 37.586178159, 40.918719012, 43.327073281,
                  48.005150881, 49.773832478])


def zeta_ratio_targets(n=10):
    """(gamma_n / gamma_1)^2 : the ratios E_n/E_1 forced by (S_V) (Remark 3.3)."""
    return (GAMMA[:n] / GAMMA[0]) ** 2


def mangoldt(nmax):
    """Array L with L[m] = Lambda(m) for 0 <= m <= nmax (sieve)."""
    L = np.zeros(nmax + 1)
    sieve = np.ones(nmax + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(nmax ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    for p in np.nonzero(sieve)[0]:
        q = int(p)
        lp = np.log(p)
        while q <= nmax:
            L[q] = lp
            q *= int(p)
    return L


_LAMBDA_CACHE = {}


def barrier_data(cls, nmax):
    """Indices k (>=2) carrying a barrier, and their unit-g0 weights w_k (g_k = g0 w_k)."""
    if cls == "log":
        # implicit: every k >= 2 with weight k^-1/2, handled inside the kernel (no arrays)
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    elif cls == "lambda":
        # one sieve, grown on demand; the kernel stops at the first k > N
        top = _LAMBDA_CACHE.get("top", 0)
        if nmax > top:
            top = max(nmax, 2 * top, 10 ** 5)
            L = mangoldt(top)
            k = np.nonzero(L)[0].astype(np.int64)
            _LAMBDA_CACHE["top"] = top
            _LAMBDA_CACHE["data"] = (k, L[k] * k.astype(np.float64) ** -0.5)
        k, w = _LAMBDA_CACHE["data"]
    else:
        raise ValueError("cls must be 'log' or 'lambda'")
    return k, w


@numba.njit(cache=True)
def _step(th, x, k, xn, g):
    th += k * (xn - x)
    m = np.floor(th / np.pi)
    r = th - m * np.pi
    if r > 1e-300 and r < np.pi - 1e-15:
        ct = np.cos(r) / np.sin(r) + g / k
        th = m * np.pi + np.arctan2(1.0, ct)
    return th


@numba.njit(cache=True)
def _count(E, x0, g0, ks, ws, N):
    """ks.shape[0] == 0 means class 'log' (barrier at every k >= 2, weight k^-1/2)."""
    k = np.sqrt(E)
    th = 0.0
    x = 0.0
    if ks.shape[0] == 0:
        for j in range(2, N + 1):
            xn = x0 * np.log(j)
            th = _step(th, x, k, xn, g0 / np.sqrt(j))
            x = xn
    else:
        for i in range(ks.shape[0]):
            j = ks[i]
            if j > N:
                break
            xn = x0 * np.log(j)
            th = _step(th, x, k, xn, g0 * ws[i])
            x = xn
    # Dirichlet end at L = x0 log N (last barrier at L contributes nothing extra)
    th += k * (x0 * np.log(N) - x)
    return int(np.floor(th / np.pi))


@numba.njit(cache=True)
def _eig(n, x0, g0, ks, ws, N, Ehi, iters):
    lo = 0.0
    hi = Ehi
    while _count(hi, x0, g0, ks, ws, N) < n:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _count(mid, x0, g0, ks, ws, N) >= n:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def count_below(E, x0, g0, cls, N):
    """Number of eigenvalues of the Dirichlet truncation H_L (L = x0 log N) below E."""
    ks, ws = barrier_data(cls, N)
    return _count(float(E), x0, g0, ks, ws, int(N))


def eigenvalue(n, x0, g0, cls, N, Ehi=1.0, iters=60):
    """n-th eigenvalue (n >= 1) of the Dirichlet truncation with N barriers' range."""
    ks, ws = barrier_data(cls, N)
    return _eig(int(n), x0, g0, ks, ws, int(N), float(Ehi), iters)


def _turning_N(E, x0, g0, fac, nmax):
    # averaged potential ~ (g0/x0) sqrt(k) reaches E at k* = (E x0 / g0)^2
    return int(min(nmax, max(1000, fac * (E * x0 / g0) ** 2)))


def eigenvalues(x0, g0, cls, nev=10, fac=20, nmax=None):
    """
    First nev eigenvalues of H (infinite lattice), via Dirichlet truncation beyond
    the classical turning point: N = fac * (E x0/g0)^2, capped at nmax.
    """
    if nmax is None:
        nmax = int(3e8) if cls == "log" else int(3e7)
    out = []
    for n in range(1, nev + 1):
        Eg = out[-1] * 1.6 if out else 5.0 / x0 ** 2
        N = _turning_N(Eg, x0, g0, fac, nmax)
        e = eigenvalue(n, x0, g0, cls, N, Ehi=Eg)
        N2 = _turning_N(e, x0, g0, fac, nmax)
        if N2 > N:
            e = eigenvalue(n, x0, g0, cls, N2, Ehi=e * 1.1)
        out.append(e)
    return np.array(out)


def ratio_table(cls, betas, nev=10, fac=(20, 60)):
    """
    Rows (beta, ratios E_n/E_1, max relative error vs zeta targets, truncation change).
    By Lemma 5.1 only beta = x0*g0 matters, so x0 = 1, g0 = beta.
    """
    tgt = zeta_ratio_targets(nev)
    rows = []
    for b in betas:
        e1 = eigenvalues(1.0, b, cls, nev, fac[0])
        e2 = eigenvalues(1.0, b, cls, nev, fac[1])
        r = e2 / e2[0]
        err = np.max(np.abs(r[1:] - tgt[1:]) / tgt[1:])
        conv = np.max(np.abs(e1 - e2) / e2)
        rows.append((b, r, err, conv, e2))
    return rows
