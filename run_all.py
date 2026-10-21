"""Run every validation script in ./validation and report PASS/FAIL."""
import glob, os, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__))
res = []
for f in sorted(glob.glob(os.path.join(here, "validation", "v*.py"))):
    t = time.time()
    p = subprocess.run([sys.executable, f], capture_output=True, text=True, cwd=os.path.join(here, "validation"))
    out = p.stdout.strip().splitlines()
    status = out[-1] if out else "NO OUTPUT"
    res.append((os.path.basename(f), status, time.time() - t))
    print(f"--- {os.path.basename(f)} ({time.time()-t:.0f} s)\n{p.stdout}{p.stderr}")
print("\nSUMMARY")
for n, s, t in res:
    print(f"{n:32s} {s[:70]:70s} {t:6.0f} s")
sys.exit(0 if all(s.startswith(("PASS", "INFO")) for _, s, _ in res) else 1)
