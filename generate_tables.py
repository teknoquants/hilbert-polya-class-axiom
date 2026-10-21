"""
Worked example (Section 5): eigenvalue-ratio test of Axiom (S_V) for the
logarithmic delta-lattice classes V_log and V_Lambda.

Reproduces Table 1 (V_log), Table 2 (V_Lambda) and Table 3 (gap comparison).

    python generate_tables.py            # full grid used in the paper (~15 min)
    python generate_tables.py --quick    # reduced grid (~1-2 min)

Outputs CSV and Markdown tables in ./results/.
"""
import argparse, os, time
import numpy as np
from hp_lattice import GAMMA, zeta_ratio_targets, ratio_table, eigenvalues

FULL_LOG = [10, 3, 1, 0.5, 0.3, 0.2, 0.125, 0.12, 0.115, 0.1, 0.05, 0.02]
FULL_LAM = [10, 3, 1, 0.5, 0.3, 0.2, 0.15, 0.14, 0.13, 0.12, 0.1]
QUICK_LOG = [3, 1, 0.5, 0.3, 0.2, 0.115]
QUICK_LAM = [3, 1, 0.5, 0.3, 0.2, 0.12]


def write(rows, cls, outdir):
    tgt = zeta_ratio_targets(10)
    csv = os.path.join(outdir, f"table_{cls}.csv")
    md = os.path.join(outdir, f"table_{cls}.md")
    with open(csv, "w") as f:
        f.write("beta," + ",".join(f"E{n}/E1" for n in range(1, 11)) + ",max_rel_err,trunc_change,E1\n")
        for b, r, err, conv, e in rows:
            f.write(f"{b}," + ",".join(f"{x:.6f}" for x in r) + f",{err:.6f},{conv:.2e},{e[0]:.10f}\n")
    with open(md, "w") as f:
        f.write("| beta | E2/E1 | E3/E1 | E4/E1 | E5/E1 | E10/E1 | max error | trunc. change |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        f.write("| zeta target | " + " | ".join(f"{tgt[i]:.3f}" for i in (1, 2, 3, 4, 9)) + " | - | - |\n")
        for b, r, err, conv, e in rows:
            f.write(f"| {b} | " + " | ".join(f"{r[i]:.3f}" for i in (1, 2, 3, 4, 9)) +
                    f" | {100*err:.0f}% | {conv:.1e} |\n")
    print(open(md).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    for cls, grid in (("log", QUICK_LOG if a.quick else FULL_LOG),
                      ("lambda", QUICK_LAM if a.quick else FULL_LAM)):
        t = time.time()
        print(f"== class V_{cls}: beta grid {grid}")
        rows = ratio_table(cls, grid)
        write(rows, cls, outdir)
        print(f"   ({time.time()-t:.0f} s)")
    # Table 3: gap comparison at the matched beta for V_log
    e = eigenvalues(1.0, 0.115, "log", 10, 60)
    lat = np.diff(np.sqrt(e / e[0]))
    zet = np.diff(GAMMA / GAMMA[0])
    with open(os.path.join(outdir, "table_gaps.md"), "w") as f:
        f.write("| gap n | " + " | ".join(str(i) for i in range(1, 10)) + " |\n|---|" + "---|" * 9 + "\n")
        f.write("| lattice beta=0.115 | " + " | ".join(f"{x:.3f}" for x in lat) + " |\n")
        f.write("| zeta zeros | " + " | ".join(f"{x:.3f}" for x in zet) + " |\n")
    print(open(os.path.join(outdir, "table_gaps.md")).read())


if __name__ == "__main__":
    main()
