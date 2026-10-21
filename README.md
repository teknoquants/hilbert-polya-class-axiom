# A Class Axiom for Hilbert–Pólya Operators — validation code

Validation code for

> R. Cooperman and A.E. Petrunina V.V., *A Class Axiom for Hilbert–Pólya Operators: Exact Spectral Theory of Logarithmic δ-Lattices and a Decisive Test*, XISOSIS Institutional Research Paper, Vienna, October 2026.

Every mathematical statement in the paper that can be checked by computation has a script here, and `generate_tables.py` is the exact program that produced the worked example (Section 5, Tables 1–3).

**Scope.** The paper does not prove the Riemann hypothesis. It proves that a class axiom (S_𝒱) would imply RH, analyses one explicit operator class rigorously, and shows numerically that this class does not satisfy the axiom. The scripts reflect that: theorems are checked for consistency, and the negative result is reproduced.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py                 # all validations, PASS/FAIL summary
python generate_tables.py --quick # worked example, reduced grid (~2 min)
python generate_tables.py         # worked example, full grid of the paper (~15 min)
```

Python ≥ 3.9 with NumPy, Numba and mpmath. The first run compiles the Numba kernels (a few seconds).

## What each script checks

| Paper result | Script | What is checked |
|---|---|---|
| §2, right side of Axiom (S_𝒱) equals ξ′/ξ; ξ(½) ≠ 0 | `validation/v1_xi_log_derivative.py` | Closed formula vs 40-digit numerical differentiation of ξ; explicit Λ-sum |
| Theorem 3.1, identity (D), Remark 3.3 | `validation/v2_conditional_theorem.py` | With E_n = γ_n²/c²: D′/D = ξ′/ξ, ξ(½)·D(s) = ξ(s), zeros at ½ ± iγ_n (300 zeros + tail estimate) |
| Proposition 4.1, tail (confinement) bound | `validation/v3_tail_bound.py` | Inequality on 300 random piecewise-linear test functions, exact integrals |
| Propositions 4.1–4.2, positivity, simplicity, N(E) bound, trace class | `validation/v4_counting_trace_class.py` | N(E) ≤ 1 + x_K√E/π; spectrum positive and simple (Sturm count jumps by exactly one at each eigenvalue); partial sums of Σ 1/E_n printed |
| Proposition 4.3, Gelfand–Yaglom transfer formula | `validation/v5_transfer_formula.py` | u_κ(L)/u_0(L) = ∏(1 + κ²/E_n(L)) and ∂_κ log u_κ(L) = 2κ Tr(H_L + κ²)⁻¹ using 20 000 eigenvalues |
| Lemma 5.1, scaling | `validation/v6_scaling_lemma.py` | E_n(x₀, g₀) = λ⁻²E_n(x₀/λ, λg₀); ratios depend only on β = x₀g₀ (both classes) |
| §5.3, decisive test | `validation/v7_ratio_test.py` | At the β matching E₂/E₁, E₃/E₁ and E₁₀/E₁ miss by more than 10% (both classes) |
| §4.4, prime-gap remark | `validation/v8_prime_gap_remark.py` | Running maximum of (q′ − q)/(√q log q) over prime powers up to 10⁷ (illustration) |
| Appendix A, reflection remark | `validation/v9_reflection_appendix.py` | R* = R, J² = I, JRJ = −R, ⟨Rv, v⟩ = 0, ⟨R²v, v⟩ = ‖Pf‖² |
| §5, Tables 1–3 (worked example) | `generate_tables.py` | Eigenvalue ratios E_n/E_1, n ≤ 10, for 𝒱_log and 𝒱_Λ over the β grid; gap comparison |

## The operator and the numerical method

H = −d²/dx² + Σ_{k≥2} g_k δ(x − x₀ log k) on L²(0, ∞), Dirichlet at 0, units ħ = 2m = 1.

- 𝒱_log: g_k = g₀ k^(−1/2)
- 𝒱_Λ: g_k = g₀ Λ(k) k^(−1/2), barriers only at prime powers

Eigenvalues of the Dirichlet truncation H_L (L = x₀ log N) come from a Prüfer angle (`hp_lattice/core.py`):
free propagation adds √E·d to θ, a barrier maps cot θ ↦ cot θ + g_k/√E, and by Sturm oscillation the number of eigenvalues below E is ⌊θ(L)/π⌋. Each E_n is found by bisection to machine precision.
The truncation is placed beyond the classical turning point, N = 20(Ex₀/g₀)², and checked against N = 60(Ex₀/g₀)².
Reported truncation changes are below 4×10⁻⁴ relative everywhere and below 4×10⁻⁵ for β ≥ 0.1.

Zero ordinates γ₁, …, γ₁₀ are hard-coded to nine decimals (LMFDB); `v2` computes 300 zeros with `mpmath.zetazero`.

## Repository layout

```
hp_lattice/core.py        operator definitions, Prüfer counting, eigenvalues, ratio tables
generate_tables.py        worked example: Tables 1–3 of the paper -> results/
validation/v1..v9_*.py    one script per checked result
run_all.py                runs all validations, prints a PASS/FAIL summary
results/                  output of generate_tables.py
```

## Status of the numerical result

Section 5 is a convergence-checked numerical result, not a computer-assisted proof. Upgrading it would need validated enclosures (interval arithmetic) for E₁, E₂, E₃ over all β and a proof that E₂/E₁ is monotone in β.

## Licence

MIT. Copyright © 2026 Roger Cooperman; A.E. Petrunina V.V.
