#!/usr/bin/env python3
"""
build_dataset.py - cleans the runs from collect.py and builds the training dataset.

Usage:
  python build_dataset.py                       # reads data/, writes dataset/
  python build_dataset.py --data data --out dataset

Reads data/run_*.csv and the matching _meta.json files. Writes to --out:
  dataset_all.csv      all cleaned rows, with inputs and force labels
  splits.csv           which runs are train, validation or test
  cleaning_report.txt  what was skipped or removed, and why
  quicklook/*.png      three quick check plots

Blocks
  The first word of each run's note sets its block:
    axial, grid, intermediate, repeat, thermal, ref, hanging
  A pose_list.csv in the data folder can override this.
  "repeat" is the second run of a pose, done on a different day.

Runs that are skipped:
  * simulated runs (collect.py --sim)
  * runs with no block in the note (for example "test" or a blank note)
  * runs that were stopped early, or have a missing or broken _meta.json
  * files that don't start with run_ (rename a bad run to bad_run_... to skip it)

Force labels (report Equation 3.6):
  alpha = rotation = x angle,  beta = tilt = y angle
  Fx = F sin(alpha) sin(beta),  Fy = F cos(alpha) sin(beta),  Fz = F cos(beta)

Needs Python 3.9+, numpy, pandas, matplotlib and scikit-learn.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# =============================== CONFIG ===============================
G = 9.80665
ALPHA_OFFSET_DEG = 0.0          # added to the x angle (check it with the sign check)
BLOCKS = ("axial", "grid", "intermediate", "repeat", "thermal", "ref", "hanging")
SKIP_UNLABELLED = True          # skip runs with no block in the note
SKIP_SIMULATED = True           # skip --sim runs
SKIP_ABORTED = True             # skip runs that were stopped early

# Cleaning rules
DROP_UNSTABLE = True            # rule 1: drop rows where the balance never settled
STIFFNESS_JUMP_FACTOR = 10.0    # rule 3: stiffness this many times the run's median means
                                # something rigid touched (normal stiffening is up to ~6x)
STIFFNESS_MIN_G = 5.0           # rule 3 ignores rows under this load
CONTACT_G = 2.0                 # above this load counts as contact
BASELINE_WEIGHT_MAX_G = 0.5     # rule 4a: zero-load row further than this from 0 g = bad tare
BASELINE_FIBRE_MAX_PM = None    # rule 4b: off. Each run uses its own zero, so drift between
                                # runs doesn't matter. Set a value in pm to turn it on.

SIGN_CHECK_MIN_N = 0.1          # sign check only uses rows above this force

# Split
TEST_BLOCKS = ("intermediate", "repeat")    # test runs
EXCLUDED_BLOCKS = ("hanging",)              # never trained on
THERMAL_TRAIN_NEEDS_TEMP_FIBRE = True       # only train on thermal runs if there's a temperature
                                            # fibre; otherwise warming looks like axial load
VAL_FRACTION = 0.15                         # share of grid runs used for validation
N_FOLDS = 5                                 # cross-validation folds (split by run)
SEED = 0
# ======================================================================


def load_runs(data_dir, log):
    csvs = sorted(glob.glob(os.path.join(data_dir, "run_*.csv")))
    if not csvs:
        sys.exit(f"No run_*.csv files found in {data_dir}")
    pose_list = {}
    pl = os.path.join(data_dir, "pose_list.csv")
    if os.path.exists(pl):
        p = pd.read_csv(pl)
        pose_list = {os.path.basename(str(f)).replace(".csv", ""): str(b).strip().lower()
                     for f, b in zip(p["filename"], p["block"])}
    frames, skipped = [], []
    run_id = 0
    for path in csvs:
        name = os.path.basename(path).replace(".csv", "")
        df = pd.read_csv(path)
        if df.empty:
            skipped.append((name, "empty file"))
            continue
        meta = {}
        mpath = path.replace(".csv", "_meta.json")
        if not os.path.exists(mpath):
            skipped.append((name, "no _meta.json file"))
            continue
        try:
            with open(mpath, encoding="utf-8") as f:
                meta = json.load(f)
        except (ValueError, OSError):
            skipped.append((name, "_meta.json is empty or unreadable (run probably interrupted)"))
            continue
        if SKIP_ABORTED and meta.get("stop_reason") == "aborted":
            skipped.append((name, f"aborted after {len(df)} rows (Ctrl-C or error)"))
            continue
        if SKIP_SIMULATED and meta.get("sim"):
            skipped.append((name, "simulated (--sim)"))
            continue
        note = str(meta.get("note", "") or "").strip()
        first = note.split()[0].lower() if note else ""
        block = pose_list.get(name, first if first in BLOCKS else None)
        if block is None:
            if SKIP_UNLABELLED:
                why = f"note '{note}' has no block" if note else "blank note"
                skipped.append((name, why))
                continue
            block = "grid"
        df["run_id"] = run_id
        df["run_name"] = name
        df["block"] = block
        df["note"] = note
        df["session"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        df["stop_reason"] = meta.get("stop_reason", "")
        frames.append(df)
        run_id += 1
    log.append(f"Run files found: {len(csvs)}, used: {len(frames)}, skipped: {len(skipped)}")
    for name, why in skipped:
        log.append(f"  skipped {name}: {why}")
    if not frames:
        sys.exit("No usable runs: every run was skipped (see the list above). Real runs need a "
                 "note starting with " + ", ".join(BLOCKS) + ".\n" + "\n".join(log))
    return pd.concat(frames, ignore_index=True)


def fibre_columns(df):
    return [c for c in df.columns if c.endswith("_nm")]


def clean(df, fibres, log):
    n0 = len(df)

    def note(rule, removed, detail=""):
        log.append(f"{rule:<45s} {removed:6d} rows removed {detail}")

    # Rule 4: bad tare (whole run). Done first, while the zero-load row is still there.
    base = df[df["step_no"] == 0].set_index("run_id")
    bad_runs = set(base.index[base["weight_g"].abs() > BASELINE_WEIGHT_MAX_G])
    if BASELINE_FIBRE_MAX_PM is not None:
        for f in fibres:
            med = base.groupby("session")[f].transform("median")
            off = (base[f] - med).abs() * 1000.0
            bad_runs |= set(base.index[off > BASELINE_FIBRE_MAX_PM])
    if bad_runs:
        names = sorted(df.loc[df["run_id"].isin(bad_runs), "run_name"].unique())
        before = len(df)
        df = df[~df["run_id"].isin(bad_runs)]
        note("Rule 4: broken baseline (whole run)", before - len(df), "\n    " + "\n    ".join(names))
    else:
        note("Rule 4: broken baseline (whole run)", 0)

    # Save each run's zero-load wavelengths before any rows are dropped.
    baselines = df[df["step_no"] == 0].set_index("run_id")[fibres]

    # Rule 1: balance didn't settle
    before = len(df)
    if DROP_UNSTABLE:
        df = df[df["weight_stable"] == 1]
    note("Rule 1: balance not stable", before - len(df))

    # Rule 2: a fibre peak went missing
    before = len(df)
    df = df.dropna(subset=fibres)
    note("Rule 2: peak dropout", before - len(df))

    # Rule 3: stiffness jump (the tip hit something rigid)
    before = len(df)
    keep = []
    for rid, run in df.groupby("run_id", sort=False):
        load = run[(run["phase"] == "load")].sort_values("step_no")
        cut_pos = None
        contact = load[load["weight_g"] > STIFFNESS_MIN_G]
        if len(contact) >= 4:
            dw = contact["weight_g"].diff()
            dp = contact["position_rev"].diff()
            k = (dw / dp).replace([np.inf, -np.inf], np.nan)
            med = k.median()
            if med and med > 0:
                jump = k > STIFFNESS_JUMP_FACTOR * med
                if jump.any():
                    cut_pos = contact.loc[jump.idxmax(), "position_rev"]
        if cut_pos is None:
            keep.append(run)
        else:
            keep.append(run[run["position_rev"] < cut_pos])
    df = pd.concat(keep) if keep else df
    note("Rule 3: stiffness jump (bottomed out)", before - len(df))

    log.append(f"{'Total':<45s} {n0 - len(df):6d} of {n0} rows removed, {len(df)} kept, "
               f"{df['run_id'].nunique()} runs")
    return df, baselines


def zero_return(raw, fibres):
    """For each run, how far each fibre ends up from where it started, compared with
    its biggest shift. A big leftover means the sensor didn't fully recover."""
    lines = ["", "Zero return (pm): last zero-load row minus first, and the largest loaded shift"]
    for rid, r in raw.groupby("run_id"):
        r = r.sort_values("step_no")
        first, last = r.iloc[0], r.iloc[-1]
        if last["phase"] != "unload" or abs(last["weight_g"]) > CONTACT_G:
            continue
        res = [(last[f] - first[f]) * 1000.0 for f in fibres]
        ld = r[r["phase"] == "load"]
        pk = ld.loc[ld["weight_g"].idxmax()] if len(ld) else first
        peak = [(pk[f] - first[f]) * 1000.0 for f in fibres]
        ratio = max(abs(v) for v in res) / max(1e-9, max(abs(v) for v in peak))
        flag = "   <-- did not recover" if ratio > 0.25 else ""
        lines.append(f"  {r['run_name'].iloc[0]:<40s} end-start {[round(v) for v in res]}  "
                     f"peak {[round(v) for v in peak]}  ({100 * ratio:.0f}%){flag}")
    return lines


def thermal_summary(raw, fibres):
    """For each thermal run: the biggest shift on each fibre, the average shift across
    the fibres, and how much axial load that would look like."""
    th = raw[raw["block"] == "thermal"]
    if th.empty:
        return []
    sens = None
    ax = raw[(raw["block"] == "axial") & (raw["phase"] == "load") & (raw["weight_g"] > 5)]
    if not ax.empty:
        ratios = []
        for _, r in ax.groupby("run_id"):
            r = r.sort_values("step_no"); b = raw[(raw["run_id"] == r["run_id"].iloc[0]) & (raw["step_no"] == 0)].iloc[0]
            top = r.iloc[-1]
            cm = sum((top[f] - b[f]) * 1000.0 for f in fibres) / len(fibres)
            ratios.append(cm / top["weight_g"])
        sens = float(np.mean(ratios))
    lines = ["", "Thermal runs (no temperature fibre: kept out of training, for characterisation):"]
    if sens:
        lines.append(f"  axial common-mode sensitivity from the axial runs: {sens:+.2f} pm/g")
    for rid, r in th.groupby("run_id"):
        r = r.sort_values("step_no"); first = r.iloc[0]
        d = {f: (r[f] - first[f]) * 1000.0 for f in fibres}
        peak = {f: d[f].iloc[d[f].abs().argmax()] for f in fibres}
        cm = (sum(d.values()) / len(fibres))
        cm_peak = cm.iloc[cm.abs().argmax()]
        eq = f", looks like {cm_peak / sens:+.1f} g of axial load" if sens else ""
        lines.append(f"  {r['run_name'].iloc[0]}: peak {[round(v) for v in peak.values()]} pm, "
                     f"common mode {cm_peak:+.0f} pm{eq}")
    return lines


def label(df, fibres, baselines):
    out = df.copy()
    # Inputs: shift from the run's own zero-load row, in pm (report Equation 4.1)
    for f in fibres:
        b = out["run_id"].map(baselines[f])
        out["dlam_" + f.replace("_nm", "")] = (out[f] - b) * 1000.0
    # Labels: report Equation 3.6
    F = out["weight_g"] * G / 1000.0
    a = np.radians(out["x_angle_deg"] + ALPHA_OFFSET_DEG)
    b_ = np.radians(out["y_angle_deg"])
    out["F"] = F
    out["alpha_deg"] = out["x_angle_deg"] + ALPHA_OFFSET_DEG
    out["beta_deg"] = out["y_angle_deg"]
    out["Fx"] = F * np.sin(a) * np.sin(b_)
    out["Fy"] = F * np.cos(a) * np.sin(b_)
    out["Fz"] = F * np.cos(b_)
    dl = [c for c in out.columns if c.startswith("dlam_")]
    return out.dropna(subset=dl)


def temperature_compensation(df, dl_cols):
    """If there's a temperature fibre, use the thermal runs to work out how much of it
    to subtract from each outer fibre (report Equation 5.2)."""
    if "dlam_temp" not in df.columns:
        return df, {}
    th = df[df["block"] == "thermal"]
    ratios = {}
    for c in dl_cols:
        if c == "dlam_temp":
            continue
        if len(th) > 5 and th["dlam_temp"].std() > 0.5:
            ratios[c] = float(np.polyfit(th["dlam_temp"], th[c], 1)[0])
        else:
            ratios[c] = 1.0     # no usable thermal runs: just subtract the temperature fibre
        df["dlamc_" + c[5:]] = df[c] - ratios[c] * df["dlam_temp"]
    return df, ratios


def make_splits(df):
    excluded = set(EXCLUDED_BLOCKS)
    if THERMAL_TRAIN_NEEDS_TEMP_FIBRE and "dlam_temp" not in df.columns:
        excluded.add("thermal")
    rng = np.random.default_rng(SEED)
    runs = df.groupby("run_id").agg(run_name=("run_name", "first"), block=("block", "first"),
                                    alpha=("alpha_deg", "first"), beta=("beta_deg", "first"),
                                    session=("session", "first")).reset_index()
    runs["split"] = "train"
    runs.loc[runs["block"].isin(TEST_BLOCKS), "split"] = "test"
    runs.loc[runs["block"].isin(excluded), "split"] = "excluded"
    grid = runs.index[(runs["split"] == "train") & (runs["block"] == "grid")].to_numpy()
    n_val = int(round(VAL_FRACTION * len(grid)))
    if n_val:
        runs.loc[rng.choice(grid, n_val, replace=False), "split"] = "val"
    # Cross-validation folds over the train and val runs (each run in one fold)
    tv = runs.index[runs["split"].isin(["train", "val"])].to_numpy()
    perm = rng.permutation(tv)
    runs["fold"] = -1
    for k, idx in enumerate(np.array_split(perm, N_FOLDS)):
        runs.loc[idx, "fold"] = k
    return runs


def quicklook(df, fibres, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    dl = ["dlam_" + f.replace("_nm", "") for f in fibres]

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in df.groupby("run_id"):
        ld = r[r["phase"] == "load"]
        ax.plot(ld["position_rev"], ld["weight_g"], lw=0.6, alpha=0.6)
    ax.set_xlabel("Slide position (rev)"); ax.set_ylabel("Balance reading (g)")
    ax.set_title("Load against position, all runs (loading path)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "qa_weight_vs_position.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, len(dl), figsize=(4 * len(dl), 3.6), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, dl):
        sc = ax.scatter(df["F"], df[c], c=df["alpha_deg"] % 360, s=3, cmap="twilight")
        ax.set_xlabel("Force magnitude (N)"); ax.set_ylabel(c + " (pm)"); ax.set_title(c)
    fig.colorbar(sc, ax=axs.tolist(), label="rotation alpha (deg)")
    fig.savefig(os.path.join(outdir, "qa_shift_vs_force.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    base = df[df["step_no"] == 0].copy()
    base["t"] = pd.to_datetime(base["timestamp"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in fibres:
        ax.plot(base["t"], (base[f] - base[f].iloc[0]) * 1000.0, ".-", label=f)
    ax.set_ylabel("Baseline drift from first run (pm)"); ax.set_title("Baseline wavelength by run start time")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "qa_baseline_drift.png"), dpi=150); plt.close(fig)


def sign_check(df, dl_cols):
    """For tilted runs, show which fibre responds most at each rotation, to check the
    angle offset."""
    lines = []
    t = df[(df["beta_deg"] >= 15) & (df["phase"] == "load") & (df["F"] > SIGN_CHECK_MIN_N)]
    for a, g in t.groupby(t["alpha_deg"] % 360):
        means = g[[c for c in dl_cols if c != "dlam_temp"]].mean()
        lines.append(f"  alpha {a:6.1f} deg: largest shift on {means.abs().idxmax()} "
                     f"({', '.join(f'{k}={v:+.1f}' for k, v in means.items())} pm)")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log = []
    raw = load_runs(args.data, log)
    fibres = fibre_columns(raw)
    log += [f"Runs used: {raw['run_id'].nunique()}, rows: {len(raw)}, fibres: {fibres}",
            "Blocks: " + ", ".join(f"{b}={n}" for b, n in raw.groupby("block")["run_id"].nunique().items()),
            ""]
    log += zero_return(raw, fibres) + thermal_summary(raw, fibres) + [""]
    df, baselines = clean(raw, fibres, log)
    df = label(df, fibres, baselines)
    dl_cols = ["dlam_" + f.replace("_nm", "") for f in fibres]
    df, ratios = temperature_compensation(df, dl_cols)
    if ratios:
        n_th = df.loc[df["block"] == "thermal", "run_id"].nunique()
        log += ["", f"Temperature compensation ratios S_T/S_T' (from {n_th} thermal runs"
                    + ("; none found, ratios set to 1.0" if n_th == 0 else "") + "):"]
        log += [f"  {k}: {v:.3f}" for k, v in ratios.items()]

    runs = make_splits(df)
    df = df.merge(runs[["run_id", "split", "fold"]], on="run_id")
    log += ["", "Split (runs): " + ", ".join(f"{s}={n}" for s, n in runs["split"].value_counts().items())]
    log += ["", "Sign check (largest responding fibre by rotation, tilt >= 15 deg):"] + sign_check(df, dl_cols)

    df.to_csv(os.path.join(args.out, "dataset_all.csv"), index=False)
    runs.to_csv(os.path.join(args.out, "splits.csv"), index=False)
    with open(os.path.join(args.out, "cleaning_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    quicklook(df, fibres, os.path.join(args.out, "quicklook"))
    print("\n".join(log))
    print(f"\nWrote {len(df)} rows to {args.out}/dataset_all.csv")


if __name__ == "__main__":
    main()