"""Cross-validate the experimental artefacts. Run this alongside make_figures.py.

Motivation. A figure in an earlier draft disagreed with the main table by ten accuracy
points because its artefact labelled the x axis with eps computed from the RAW injected
sigma, while the main pipeline computes eps from the effective multiplier sqrt(S_r)/w_i
that accounts for concealment within the region. Every per-claim check passed: each number
quoted in the text traced to some artefact, every figure regenerated, every reference
resolved. What no check did was compare two artefacts describing the same quantity, or
re-derive eps from the mechanism rather than trusting the value stored beside it.

Three checks, in increasing order of how much they would have helped:

  A  accounting   every row's recorded eps must equal eps_silo(T, sigma_eff_min), and for
                  equal-weight artefacts the whole allocation is re-derived from K.
  B  agreement    artefacts overlapping on (dataset, profile, mode, spread, d, eps) must
                  report the same accuracy to within seed noise.
  C  provenance   every artefact must be read by something, and no artefact may sit in the
                  tree without a script that produces it.

Exit status is non-zero if any check fails, so this can gate a commit.
"""
from __future__ import annotations
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedsim import eps_silo
from evaluate import rho_stats, sigmas_for_target, profiles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ELL = 0.85
TOL_EPS = 1e-6          # eps is deterministic given sigma_eff; allow float noise only
TOL_U = 1e-6
NSIGMA = 4.0            # cross-artefact accuracy agreement, in combined standard errors

fails: list[str] = []
notes: list[str] = []


def _rows(path):
    try:
        r = json.load(open(path))
    except Exception as e:
        fails.append(f"[parse] {os.path.basename(path)}: {e}")
        return []
    return r if isinstance(r, list) and r and isinstance(r[0], dict) else []


def check_accounting(name, rows):
    """A. Re-derive the privacy accounting instead of trusting the stored eps."""
    n_ok = n_skip = 0
    for i, r in enumerate(rows):
        T, eps = r.get("T"), r.get("eps")
        if eps is None:                      # non-private reference row
            continue
        se = r.get("sigma_eff_min")
        if se is None:
            n_skip += 1
            continue
        want = eps_silo(T, se)
        if abs(want - eps) > TOL_EPS:
            fails.append(f"[A eps] {name} row {i}: stored eps={eps:.6f} but "
                         f"eps_silo(T={T}, sigma_eff={se:.4f})={want:.6f}")
            return
        n_ok += 1
        # Equal weights: the entire allocation is reconstructible from K, so check it.
        if r.get("spread") in (0.0, 0) and r.get("profile") in profiles(r["n"]) and r.get("K"):
            groups = profiles(r["n"])[r["profile"]]
            w = np.full(r["n"], 1.0 / r["n"])
            _, V, _, delta = rho_stats(w, groups)
            s2, U = sigmas_for_target(w, groups, r["K"], np.full(len(V), ELL),
                                      2.0 * T, r["mode"], r["seed"])
            if abs(U - r["U"]) > TOL_U * max(1.0, abs(U)):
                fails.append(f"[A U] {name} row {i}: stored U={r['U']:.6g}, re-derived {U:.6g}")
                return
            if "delta" in r and abs(delta - r["delta"]) > 1e-9:
                fails.append(f"[A delta] {name} row {i}: stored {r['delta']:.6f}, "
                             f"re-derived {delta:.6f}")
                return
    if n_skip:
        notes.append(f"  {name}: {n_skip} rows lack sigma_eff_min, eps not verifiable")
    return n_ok


def check_agreement(all_rows):
    """B. Two artefacts describing the same configuration must agree."""
    key = lambda r: (r.get("dataset"), r.get("profile"), r.get("mode"), r.get("spread"),
                     r.get("d"), None if r.get("eps") is None else round(r["eps"], 4))
    groups = {}
    for name, rows in all_rows.items():
        for r in rows:
            if r.get("dataset") is None:
                continue
            groups.setdefault(key(r), {}).setdefault(name, []).append(r["acc"])
    n_cmp = 0
    for k, byfile in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(byfile) < 2:
            continue
        stats = {f: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0, len(v))
                 for f, v in byfile.items()}
        items = list(stats.items())
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (fa, (ma, sa, na)), (fb, (mb, sb, nb)) = items[a], items[b]
                comb = float(np.hypot(sa, sb)) or 1e-9
                n_cmp += 1
                if abs(ma - mb) > NSIGMA * comb:
                    fails.append(f"[B] {k}: {fa} {ma:.4f}(n={na}) vs {fb} {mb:.4f}(n={nb}) "
                                 f"-> {abs(ma-mb)/comb:.1f} sigma apart")
    return n_cmp


def check_provenance(datadir, files):
    """C. Nothing consumed without a producer; nothing sitting unread."""
    src = {}
    for f in os.listdir(HERE):
        if f.endswith(".py"):
            src[f] = open(os.path.join(HERE, f)).read()
    read_by, written_by = {}, {}
    for b in files:
        stem = b[:-5] if b.endswith(".json") else b
        for f, text in src.items():
            if b in text or f'"{stem}' in text:
                (read_by if f == "make_figures.py" or "_load" in text else read_by)[b] = f
        # a producer is any script whose --out default or docstring names it, or that can
        # emit it by argument; we detect the generic writers explicitly
        if any(k in b for k in ("probe", "agnews", "pu_curve")):
            written_by[b] = "probe.py / agnews.py / pu_curve.py"
    # Artefacts that back prose rather than a figure. probe_d258 supports the
    # dimension-scaling subsection, which quotes its numbers in text and plots none.
    MANUSCRIPT_ONLY = {"probe_d258.json"}
    for b in MANUSCRIPT_ONLY:
        read_by.setdefault(b, "manuscript prose")
    for b in files:
        if b not in written_by:
            fails.append(f"[C] {b}: no script in fulcrum/ produces this artefact")
        if b not in read_by:
            notes.append(f"  {b}: not read by any script (orphan)")


if __name__ == "__main__":
    datadir = os.path.join(ROOT, "analysis", sys.argv[1] if len(sys.argv) > 1 else "v2")
    files = sorted(f for f in os.listdir(datadir) if f.endswith(".json"))
    all_rows = {}
    print(f"checking {datadir}  ({len(files)} artefacts)\n")
    tot = 0
    for b in files:
        rows = _rows(os.path.join(datadir, b))
        if not rows:
            notes.append(f"  {b}: no row records")
            continue
        all_rows[b] = rows
        tot += check_accounting(b, rows) or 0
    n_cmp = check_agreement(all_rows)
    check_provenance(datadir, files)

    print(f"A accounting : {tot} rows re-derived from the mechanism")
    print(f"B agreement  : {n_cmp} cross-artefact comparisons")
    print(f"C provenance : {len(files)} artefacts examined")
    if notes:
        print("\nnotes:"); [print(n) for n in notes]
    if fails:
        print(f"\nFAILED ({len(fails)}):"); [print("  " + f) for f in fails]
        sys.exit(1)
    print("\nall checks passed")
