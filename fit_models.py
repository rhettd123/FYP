#!/usr/bin/env python3
"""
fit_models.py - fits the calibration models to the dataset from build_dataset.py
and makes the results tables and figures.

Usage:
  python fit_models.py                        # reads dataset/, writes results/
  python fit_models.py --quick                # faster run, for testing
  python fit_models.py --predict file.csv     # forces from a CSV of fibre shifts (pm)

Models (all trained and tested on the same runs):
  Linear LSQ     straight-line fit, F = K x shifts + c
  Ridge          linear fit with a small penalty to keep it stable
  Quadratic      also uses the squares and products of the shifts
  Neural network small network with two hidden layers, averaged over several seeds
  GP             Gaussian process, which also gives an error bar on each prediction

Results (report Section 4.6): errors on each axis and in direction, broken down by
tilt, load, loading or unloading, and block, plus cross-validation by run and a
learning curve.

Saved to results/: the metrics CSVs, summary.txt, figures/*.png and models.joblib
(which live_force.py loads).

Needs Python 3.9+, numpy, pandas, matplotlib, scikit-learn and joblib.
"""

import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# =============================== CONFIG ===============================
AXES = ("Fx", "Fy", "Fz")
PRIOR_RMSE = {"Fx": 0.0688, "Fy": 0.0688, "Fz": 0.233}   # 2025 sensor's training error [9]

# Inputs: "auto" uses f1 to f3, plus the temperature fibre if there is one.
INPUT_SET = "auto"           # or "outer", "outer+temp", "compensated"

# Ridge: penalties tried on the validation runs
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
QUAD_ALPHA = 1e-4

# Neural network (report Section 4.5)
NN_WIDTHS = (16, 32, 64)
NN_ACTIVATIONS = ("tanh", "relu")
NN_L2 = (0.0, 1e-4, 1e-3)
NN_LR = 1e-3
NN_BATCH = 64
NN_MAX_EPOCHS = 3000
NN_PATIENCE = 50             # stop after this many epochs with no improvement
NN_SEEDS = 5                 # networks averaged in the final model

# Gaussian process
GP_MAX_ROWS = 1500           # rows used to train the GP (it gets slow with more)
GP_RESTARTS = 2

# Cross-validation and learning curve
CV_NN_SEEDS = 3
LC_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
LC_REPEATS = 3
LC_INCLUDE_GP = True

ANGULAR_MIN_N = 0.1          # only measure the angle error above this force
SEED = 0
# ======================================================================

MODEL_ORDER = ("Linear LSQ", "Ridge", "Quadratic", "Neural network", "GP")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ models
class LinearLSQ:
    """Least-squares straight-line fit, F = K x shifts + c (N per pm)."""
    name = "Linear LSQ"

    def fit(self, X, Y, Xv=None, Yv=None):
        Xa = np.hstack([X, np.ones((len(X), 1))])
        self.KT, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
        self.cond = float(np.linalg.cond(Xa))
        return self

    def predict(self, X):
        return np.hstack([X, np.ones((len(X), 1))]) @ self.KT

    @property
    def K(self):
        return self.KT[:-1].T       # 3 x m

    @property
    def offset(self):
        return self.KT[-1]


class RidgeModel:
    name = "Ridge"

    def __init__(self, alpha=None):
        self.alpha = alpha

    def _fit(self, X, Y, alpha):
        self.sx, self.sy = StandardScaler().fit(X), StandardScaler().fit(Y)
        self.m = Ridge(alpha=alpha).fit(self.sx.transform(X), self.sy.transform(Y))
        return self

    def fit(self, X, Y, Xv=None, Yv=None):
        if self.alpha is None:
            if Xv is None:
                raise ValueError("Ridge needs validation data to choose alpha")
            scores = {}
            for a in RIDGE_ALPHAS:
                self._fit(X, Y, a)
                scores[a] = mse_std(self, Xv, Yv)
            self.alpha = min(scores, key=scores.get)
            self.alpha_scores = scores
        return self._fit(X, Y, self.alpha)

    def predict(self, X):
        return self.sy.inverse_transform(self.m.predict(self.sx.transform(X)))


class QuadraticModel:
    name = "Quadratic"

    def fit(self, X, Y, Xv=None, Yv=None):
        self.pf = PolynomialFeatures(degree=2, include_bias=False)
        P = self.pf.fit_transform(X)
        self.sx, self.sy = StandardScaler().fit(P), StandardScaler().fit(Y)
        self.m = Ridge(alpha=QUAD_ALPHA).fit(self.sx.transform(P), self.sy.transform(Y))
        return self

    def predict(self, X):
        P = self.sx.transform(self.pf.transform(X))
        return self.sy.inverse_transform(self.m.predict(P))


def train_mlp(X, Y, Xv, Yv, width, act, l2, seed, max_epochs=NN_MAX_EPOCHS):
    """Train one network, stopping once it stops improving on the validation runs.
    Returns (model, scalers, history)."""
    sx, sy = StandardScaler().fit(X), StandardScaler().fit(Y)
    Xs, Ys, Xvs, Yvs = sx.transform(X), sy.transform(Y), sx.transform(Xv), sy.transform(Yv)
    m = MLPRegressor(hidden_layer_sizes=(width, width), activation=act, solver="adam",
                     alpha=l2, learning_rate_init=NN_LR, batch_size=min(NN_BATCH, len(X)),
                     random_state=seed, shuffle=True)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        m.partial_fit(Xs, Ys)
        tr = float(np.mean((m.predict(Xs) - Ys) ** 2))
        va = float(np.mean((m.predict(Xvs) - Yvs) ** 2))
        hist.append((ep + 1, tr, va))
        if va < best - 1e-7:
            best, wait = va, 0
            best_state = (copy.deepcopy(m.coefs_), copy.deepcopy(m.intercepts_), ep + 1)
        else:
            wait += 1
            if wait >= NN_PATIENCE:
                break
    m.coefs_, m.intercepts_, best_ep = best_state
    return m, (sx, sy), {"history": hist, "best_epoch": best_ep, "best_val": best}


class NNEnsemble:
    name = "Neural network"

    def __init__(self, width=None, act=None, l2=None, seeds=NN_SEEDS, max_epochs=NN_MAX_EPOCHS):
        self.width, self.act, self.l2 = width, act, l2
        self.seeds, self.max_epochs = seeds, max_epochs

    def fit(self, X, Y, Xv, Yv):
        if self.width is None:
            self.search = []
            for w in NN_WIDTHS:
                for a in NN_ACTIVATIONS:
                    for l2 in NN_L2:
                        _, _, info = train_mlp(X, Y, Xv, Yv, w, a, l2, SEED, self.max_epochs)
                        self.search.append({"width": w, "activation": a, "l2": l2,
                                            "val_mse_std": info["best_val"], "epochs": info["best_epoch"]})
                        log(f"    NN search width={w:<3d} {a:<4s} l2={l2:<6g} val={info['best_val']:.4f} "
                            f"epochs={info['best_epoch']}")
            # pick the smallest network within 2% of the best
            s = pd.DataFrame(self.search)
            tol = s["val_mse_std"].min() * 1.02
            pick = s[s["val_mse_std"] <= tol].sort_values(["width", "val_mse_std"]).iloc[0]
            self.width, self.act, self.l2 = int(pick["width"]), pick["activation"], float(pick["l2"])
        self.members, self.infos = [], []
        for k in range(self.seeds):
            m, sc, info = train_mlp(X, Y, Xv, Yv, self.width, self.act, self.l2, SEED + k, self.max_epochs)
            self.members.append((m, sc))
            self.infos.append(info)
        return self

    def predict_members(self, X):
        return np.stack([sc[1].inverse_transform(m.predict(sc[0].transform(X))) for m, sc in self.members])

    def predict(self, X):
        return self.predict_members(X).mean(axis=0)


class GPModel:
    name = "GP"

    def __init__(self, max_rows=GP_MAX_ROWS, restarts=GP_RESTARTS):
        self.max_rows, self.restarts = max_rows, restarts

    def fit(self, X, Y, Xv=None, Yv=None):
        rng = np.random.default_rng(SEED)
        if len(X) > self.max_rows:
            idx = rng.choice(len(X), self.max_rows, replace=False)
            X, Y = X[idx], Y[idx]
        self.n_train = len(X)
        self.sx, self.sy = StandardScaler().fit(X), StandardScaler().fit(Y)
        d = X.shape[1]
        kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.ones(d), (1e-2, 1e3))
                  + WhiteKernel(1e-2, (1e-6, 1.0)))
        self.m = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                          n_restarts_optimizer=self.restarts, random_state=SEED)
        self.m.fit(self.sx.transform(X), self.sy.transform(Y))
        return self

    def predict(self, X, return_std=False):
        mu, sd = self.m.predict(self.sx.transform(X), return_std=True)
        mu = self.sy.inverse_transform(mu)
        if sd.ndim == 1:
            sd = np.repeat(sd[:, None], mu.shape[1], axis=1)
        sd = sd * self.sy.scale_
        return (mu, sd) if return_std else mu


def mse_std(model, X, Y):
    sy = StandardScaler().fit(Y)
    return float(np.mean((sy.transform(model.predict(X)) - sy.transform(Y)) ** 2))


# ----------------------------------------------------------------- metrics
def metrics(Y, P, fs):
    out = {}
    for j, ax in enumerate(AXES):
        e = P[:, j] - Y[:, j]
        ss_tot = np.sum((Y[:, j] - Y[:, j].mean()) ** 2)
        out[f"RMSE_{ax}"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"MAE_{ax}"] = float(np.mean(np.abs(e)))
        out[f"Max_{ax}"] = float(np.max(np.abs(e)))
        out[f"nRMSE_{ax}_%"] = 100.0 * out[f"RMSE_{ax}"] / fs[ax]
        out[f"R2_{ax}"] = float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else np.nan
    nt, npred = np.linalg.norm(Y, axis=1), np.linalg.norm(P, axis=1)
    out["RMSE_magnitude"] = float(np.sqrt(np.mean((npred - nt) ** 2)))
    ok = nt > ANGULAR_MIN_N
    if ok.any():
        c = np.sum(P[ok] * Y[ok], axis=1) / (npred[ok] * nt[ok] + 1e-12)
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        out["Angular_mean_deg"] = float(ang.mean())
        out["Angular_median_deg"] = float(np.median(ang))
    out["n_rows"] = int(len(Y))
    return out


def breakdowns(test, P_by_model, fs):
    rows = []
    t = test.copy()
    fmax = test["F"].max() if len(test) else 1.0
    t["load_band"] = pd.cut(t["F"] / fmax, [-0.01, 0.25, 0.5, 0.75, 1.01],
                            labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    t["tilt"] = t["beta_deg"].round(1)
    for by in ("tilt", "load_band", "phase", "block"):
        for key, g in t.groupby(by, observed=True):
            idx = g.index.to_numpy()
            for name, P in P_by_model.items():
                r = metrics(g[list(AXES)].to_numpy(), P[t.index.get_indexer(idx)], fs)
                rows.append({"breakdown": by, "group": key, "model": name,
                             **{k: r[k] for k in r if k.startswith(("RMSE", "Angular", "n_rows"))}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- data
def choose_inputs(df):
    outer = sorted(c for c in df.columns if c.startswith("dlam_f"))
    has_temp = "dlam_temp" in df.columns
    comp = sorted(c for c in df.columns if c.startswith("dlamc_"))
    mode = INPUT_SET
    if mode == "auto":
        mode = "outer+temp" if has_temp else "outer"
    if mode == "outer":
        return outer
    if mode == "outer+temp":
        return outer + ["dlam_temp"]
    if mode == "compensated":
        return comp
    raise ValueError(INPUT_SET)


def xy(d, cols):
    return d[cols].to_numpy(float), d[list(AXES)].to_numpy(float)


# ------------------------------------------------------------------- plots
def plots(res, test, preds, nn, lin, gp_sd, lc, outdir, cols, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    Y = test[list(AXES)].to_numpy()
    shown = [m for m in ("Linear LSQ", "Neural network", "GP") if m in preds]
    mk = {"Linear LSQ": ("o", "#7f7f7f"), "Neural network": ("^", "#1f77b4"), "GP": ("s", "#d62728"),
          "Ridge": ("v", "#2ca02c"), "Quadratic": ("D", "#ff7f0e")}

    # 1 predicted vs true
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for j, ax in enumerate(AXES):
        a = axs[j]
        for m in shown:
            a.scatter(Y[:, j], preds[m][:, j], s=5, marker=mk[m][0], c=mk[m][1], alpha=0.5, label=m)
        lo, hi = Y[:, j].min(), Y[:, j].max()
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set_xlabel(f"True {ax} (N)"); a.set_ylabel(f"Predicted {ax} (N)"); a.set_title(ax)
    axs[0].legend(markerscale=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/pred_vs_true.png", dpi=170); plt.close(fig)

    # 2 residual vs true
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, ax in enumerate(AXES):
        for m in shown:
            axs[j].scatter(Y[:, j], preds[m][:, j] - Y[:, j], s=5, marker=mk[m][0], c=mk[m][1], alpha=0.5, label=m)
        axs[j].axhline(0, c="k", lw=0.8); axs[j].set_xlabel(f"True {ax} (N)"); axs[j].set_ylabel("Residual (N)")
        axs[j].set_title(ax)
    axs[0].legend(markerscale=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/residual_vs_true.png", dpi=170); plt.close(fig)

    # 3 residual vs tilt, coloured by phase
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for i, m in enumerate(["Linear LSQ", "Neural network"]):
        if m not in preds:
            continue
        r = preds[m][:, 2] - Y[:, 2]
        for ph, c in (("load", "#1f77b4"), ("unload", "#ff7f0e")):
            s = test["phase"].to_numpy() == ph
            axs[i].scatter(test["beta_deg"].to_numpy()[s] + (0.4 if ph == "unload" else 0), r[s], s=5, c=c, label=ph, alpha=0.6)
        axs[i].axhline(0, c="k", lw=0.8); axs[i].set_title(f"{m}: Fz residual")
        axs[i].set_xlabel("Tilt beta (deg)"); axs[i].set_ylabel("Residual (N)")
    axs[0].legend(markerscale=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/residual_vs_tilt.png", dpi=170); plt.close(fig)

    # 4 axial RMSE vs tilt for each model, with the prior sensor for reference
    fig, a = plt.subplots(figsize=(6, 3.8))
    tilts = sorted(test["beta_deg"].round(1).unique())
    for m in preds:
        vals = []
        for t in tilts:
            s = test["beta_deg"].round(1).to_numpy() == t
            vals.append(np.sqrt(np.mean((preds[m][s, 2] - Y[s, 2]) ** 2)))
        a.plot(tilts, vals, marker=mk.get(m, ("o", None))[0], label=m)
    a.axhline(PRIOR_RMSE["Fz"], c="k", ls="--", lw=0.8, label="Prior sensor (training error)")
    a.set_xlabel("Tilt beta (deg)"); a.set_ylabel("Fz RMSE (N)"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/axial_rmse_vs_tilt.png", dpi=170); plt.close(fig)

    # 5 NN training curve (first network)
    h = np.array(nn.infos[0]["history"])
    fig, a = plt.subplots(figsize=(6, 3.6))
    a.plot(h[:, 0], h[:, 1], label="Training"); a.plot(h[:, 0], h[:, 2], label="Validation runs")
    a.axvline(nn.infos[0]["best_epoch"], c="k", ls="--", lw=0.8, label="Early stopping point")
    a.set_yscale("log"); a.set_xlabel("Epoch"); a.set_ylabel("MSE (standardised)"); a.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/nn_training_curve.png", dpi=170); plt.close(fig)

    # 6 calibration matrix
    K = lin.K
    fig, a = plt.subplots(figsize=(1.4 + 1.2 * K.shape[1], 3))
    v = np.abs(K).max()
    a.imshow(K, cmap="RdBu_r", vmin=-v, vmax=v)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            a.text(j, i, f"{K[i, j]*1000:.2f}", ha="center", va="center", fontsize=9)
    a.set_xticks(range(len(cols))); a.set_xticklabels([c.replace("dlam_", "") for c in cols])
    a.set_yticks(range(3)); a.set_yticklabels(AXES)
    a.set_title("Calibration matrix K (mN per pm)")
    fig.tight_layout(); fig.savefig(f"{outdir}/calibration_matrix.png", dpi=170); plt.close(fig)

    # 7 GP prediction of Fz with its ±2σ band
    if gp_sd is not None:
        fig, a = plt.subplots(figsize=(7, 3.8))
        o = np.argsort(Y[:, 2])
        a.fill_between(np.arange(len(o)), preds["GP"][o, 2] - 2 * gp_sd[o, 2], preds["GP"][o, 2] + 2 * gp_sd[o, 2],
                       color="#d62728", alpha=0.2, label="GP ±2σ")
        a.plot(np.arange(len(o)), preds["GP"][o, 2], c="#d62728", lw=0.8, label="GP prediction")
        a.plot(np.arange(len(o)), Y[o, 2], "k.", ms=2, label="True Fz")
        a.set_xlabel("Test rows, sorted by true Fz"); a.set_ylabel("Fz (N)"); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{outdir}/gp_uncertainty.png", dpi=170); plt.close(fig)

    # 8 learning curve
    if lc is not None and len(lc):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for m, g in lc.groupby("model"):
            g2 = g.groupby("n_runs").agg(fz=("RMSE_Fz", "mean"), fzs=("RMSE_Fz", "std"),
                                          lat=("RMSE_lat", "mean"), lats=("RMSE_lat", "std")).reset_index()
            g2[["fzs", "lats"]] = g2[["fzs", "lats"]].fillna(0.0)   # only one repeat at full data, so no spread
            axs[0].errorbar(g2["n_runs"], g2["fz"], yerr=g2["fzs"], marker=mk.get(m, ("o",))[0], capsize=3, label=m)
            axs[1].errorbar(g2["n_runs"], g2["lat"], yerr=g2["lats"], marker=mk.get(m, ("o",))[0], capsize=3, label=m)
        axs[0].set_title("Axial (Fz) test RMSE"); axs[1].set_title("Transverse test RMSE (mean of Fx, Fy)")
        for a in axs:
            a.set_xlabel("Training runs used"); a.set_ylabel("RMSE (N)")
        axs[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{outdir}/learning_curve.png", dpi=170); plt.close(fig)

    # 9 raw response: shift vs force for each fibre at three tilts
    tilts_all = sorted(full["beta_deg"].round(1).unique())
    pick = sorted({tilts_all[0], tilts_all[len(tilts_all) // 2], tilts_all[-1]})
    outer = [c for c in cols if c.startswith("dlam_f")]
    fig, axs = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for a, t in zip(axs, pick):
        s = full[full["beta_deg"].round(1) == t]
        run = s[s["run_id"] == s["run_id"].iloc[0]]
        for c in outer:
            for ph, ls in (("load", "-"), ("unload", "--")):
                r = run[run["phase"] == ph].sort_values("F")
                a.plot(r["F"], r[c], ls, marker=".", ms=3, label=f"{c[5:]} {ph}")
        a.set_title(f"beta = {t:g} deg, alpha = {run['alpha_deg'].iloc[0]:g} deg")
        a.set_xlabel("Force magnitude (N)")
    axs[0].set_ylabel("Wavelength shift (pm)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir}/raw_response.png", dpi=170); plt.close(fig)


# -------------------------------------------------------------------- main
def fit_all(tr, va, cols, nn_cfg=None, nn_seeds=NN_SEEDS, ridge_alpha=None, include_gp=True,
            max_epochs=NN_MAX_EPOCHS, gp_rows=GP_MAX_ROWS, gp_restarts=GP_RESTARTS):
    X, Y = xy(tr, cols)
    Xv, Yv = xy(va, cols)
    models = {}
    models["Linear LSQ"] = LinearLSQ().fit(X, Y)
    models["Ridge"] = RidgeModel(ridge_alpha).fit(X, Y, Xv, Yv)
    models["Quadratic"] = QuadraticModel().fit(X, Y)
    w, a, l2 = nn_cfg if nn_cfg else (None, None, None)
    models["Neural network"] = NNEnsemble(w, a, l2, seeds=nn_seeds, max_epochs=max_epochs).fit(X, Y, Xv, Yv)
    if include_gp:
        models["GP"] = GPModel(gp_rows, gp_restarts).fit(np.vstack([X, Xv]), np.vstack([Y, Yv]))
    return models


def main():
    global NN_SEEDS, CV_NN_SEEDS, LC_REPEATS, LC_INCLUDE_GP, INPUT_SET
    ap = argparse.ArgumentParser(description="Fit and compare calibration models")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="fewer seeds, no GP in CV or learning curve")
    ap.add_argument("--no-gp", action="store_true", help="leave the Gaussian process out entirely")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--no-lc", action="store_true")
    ap.add_argument("--inputs", choices=["auto", "outer", "outer+temp", "compensated"], default=None,
                    help="input set (default INPUT_SET in the config block); run once with 'outer' and once "
                         "with 'outer+temp' to measure what the reference fibre adds")
    ap.add_argument("--predict", help="CSV with dlam_* columns in pm; writes <name>_forces.csv")
    args = ap.parse_args()
    if args.inputs:
        INPUT_SET = args.inputs

    if args.predict:
        import joblib
        saved = joblib.load(os.path.join(args.out, "models.joblib"))
        d = pd.read_csv(args.predict)
        X = d[saved["inputs"]].to_numpy(float)
        for name, m in saved["models"].items():
            P = m.predict(X)
            for j, ax in enumerate(AXES):
                d[f"{ax}_{name.replace(' ', '_')}"] = P[:, j]
        out = args.predict.replace(".csv", "_forces.csv")
        d.to_csv(out, index=False)
        log(f"Wrote {out}")
        return

    if args.quick:
        NN_SEEDS, CV_NN_SEEDS, LC_REPEATS, LC_INCLUDE_GP = 2, 1, 2, False
    t0 = time.time()
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.dataset, "dataset_all.csv"))
    cols = choose_inputs(df)
    tr, va, te = df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]
    if te.empty:
        sys.exit("No test runs. Label some runs 'intermediate' or 'repeat' (see build_dataset.py).")
    fs = {ax: float(df[ax].abs().max()) or 1.0 for ax in AXES}
    log(f"Inputs {cols}; runs train/val/test = {tr.run_id.nunique()}/{va.run_id.nunique()}/{te.run_id.nunique()}, "
        f"rows {len(tr)}/{len(va)}/{len(te)}")

    # ---------------- main fit on the fixed split
    log("Fitting models on the training runs ...")
    use_gp = not args.no_gp
    models = fit_all(tr, va, cols, include_gp=use_gp)
    nn, lin = models["Neural network"], models["Linear LSQ"]
    nn_cfg = (nn.width, nn.act, nn.l2)
    Xt, Yt = xy(te, cols)
    preds, gp_sd = {}, None
    for name, m in models.items():
        if name == "GP":
            preds[name], gp_sd = m.predict(Xt, return_std=True)
        else:
            preds[name] = m.predict(Xt)
    mt = pd.DataFrame([{"model": n, **metrics(Yt, preds[n], fs)} for n in preds]).set_index("model")
    # how much single networks vary between seeds
    member = nn.predict_members(Xt)
    seed_rmse = np.array([[np.sqrt(np.mean((member[k][:, j] - Yt[:, j]) ** 2)) for j in range(3)]
                          for k in range(len(member))])
    mt.to_csv(os.path.join(args.out, "metrics_test.csv"))
    bd = breakdowns(te.reset_index(drop=True), preds, fs)
    bd.to_csv(os.path.join(args.out, "breakdowns.csv"), index=False)

    Kdf = pd.DataFrame(lin.K, index=AXES, columns=cols)
    Kdf["offset_N"] = lin.offset
    Kdf.to_csv(os.path.join(args.out, "calibration_matrix.csv"))

    cov = None
    if gp_sd is not None:
        inside = np.abs(preds["GP"] - Yt) <= 2 * gp_sd
        cov = pd.DataFrame({"axis": AXES, "coverage_2sigma_%": 100 * inside.mean(axis=0),
                            "mean_sigma_N": gp_sd.mean(axis=0)})
        cov.to_csv(os.path.join(args.out, "gp_coverage.csv"), index=False)

    # hanging-mass runs, if there are any
    ex = df[df["block"] == "hanging"]
    hang = None
    if len(ex):
        Xh, Yh = xy(ex, cols)
        hang = pd.DataFrame([{"model": n, **metrics(Yh, m.predict(Xh), fs)} for n, m in models.items()])
        hang.to_csv(os.path.join(args.out, "metrics_hanging_mass.csv"), index=False)

    # ---------------- cross-validation, split by run
    cv = None
    if not args.no_cv:
        log("Grouped k-fold cross-validation ...")
        tv = df[df["split"].isin(["train", "val"])]
        folds = sorted(tv["fold"].unique())
        rows = []
        for k in folds:
            inner = folds[(folds.index(k) + 1) % len(folds)]
            ftr = tv[(tv["fold"] != k) & (tv["fold"] != inner)]
            fva = tv[tv["fold"] == inner]
            fte = tv[tv["fold"] == k]
            ms = fit_all(ftr, fva, cols, nn_cfg=nn_cfg, nn_seeds=CV_NN_SEEDS,
                         ridge_alpha=models["Ridge"].alpha, include_gp=use_gp and LC_INCLUDE_GP,
                         gp_restarts=0)
            Xf, Yf = xy(fte, cols)
            for n, m in ms.items():
                rows.append({"fold": k, "model": n, **metrics(Yf, m.predict(Xf), fs)})
            log(f"  fold {k} done")
        cvr = pd.DataFrame(rows)
        cvr.to_csv(os.path.join(args.out, "metrics_cv_folds.csv"), index=False)
        keep = [c for c in cvr.columns if c.startswith(("RMSE", "nRMSE", "Angular"))]
        cv = cvr.groupby("model")[keep].agg(["mean", "std"])
        cv.to_csv(os.path.join(args.out, "metrics_cv.csv"))

    # ---------------- learning curve on the test set
    lc = None
    if not args.no_lc:
        log("Learning curve ...")
        rng = np.random.default_rng(SEED)
        tr_runs = tr["run_id"].unique()
        rows = []
        for frac in LC_FRACTIONS:
            n = max(2, int(round(frac * len(tr_runs))))
            reps = 1 if n == len(tr_runs) else LC_REPEATS
            for r in range(reps):
                pick = rng.choice(tr_runs, n, replace=False)
                sub = tr[tr["run_id"].isin(pick)]
                ms = fit_all(sub, va, cols, nn_cfg=nn_cfg, nn_seeds=1, ridge_alpha=models["Ridge"].alpha,
                             include_gp=use_gp and LC_INCLUDE_GP, gp_restarts=0)
                for name, m in ms.items():
                    mm = metrics(Yt, m.predict(Xt), fs)
                    rows.append({"model": name, "fraction": frac, "n_runs": n, "n_rows": len(sub), "repeat": r,
                                 "RMSE_Fx": mm["RMSE_Fx"], "RMSE_Fy": mm["RMSE_Fy"], "RMSE_Fz": mm["RMSE_Fz"],
                                 "RMSE_lat": 0.5 * (mm["RMSE_Fx"] + mm["RMSE_Fy"])})
            log(f"  {n} runs done")
        lc = pd.DataFrame(rows)
        lc.to_csv(os.path.join(args.out, "learning_curve.csv"), index=False)

    # ---------------- figures, saved models, summary
    plots(mt, te.reset_index(drop=True), preds, nn, lin, gp_sd, lc, figdir, cols, df)
    try:
        import joblib
        joblib.dump({"models": models, "inputs": cols, "nn_config": nn_cfg}, os.path.join(args.out, "models.joblib"))
    except Exception as e:
        log(f"Could not save models: {e}")

    S = []
    S.append("CALIBRATION RESULTS SUMMARY")
    S.append(f"Inputs: {', '.join(cols)}")
    S.append(f"Runs (rows): train {tr.run_id.nunique()} ({len(tr)}), validation {va.run_id.nunique()} ({len(va)}), "
             f"test {te.run_id.nunique()} ({len(te)})")
    S.append(f"Full scale used for nRMSE: " + ", ".join(f"{a} {fs[a]:.3f} N" for a in AXES))
    S.append("")
    S.append("Test-set RMSE (N) [report Table 6.2]")
    S.append(f"  {'Model':<16s}{'Fx':>9s}{'Fy':>9s}{'Fz':>9s}{'nRMSE Fz':>10s}{'R2 Fz':>8s}{'Angle':>8s}")
    S.append(f"  {'Prior sensor*':<16s}{0.0688:9.4f}{0.0688:9.4f}{0.233:9.4f}")
    for n in [m for m in MODEL_ORDER if m in mt.index]:
        r = mt.loc[n]
        S.append(f"  {n:<16s}{r.RMSE_Fx:9.4f}{r.RMSE_Fy:9.4f}{r.RMSE_Fz:9.4f}{r['nRMSE_Fz_%']:9.1f}%"
                 f"{r.R2_Fz:8.3f}{r.get('Angular_mean_deg', np.nan):7.1f}°")
    S.append("  * prior sensor figures are training errors, not test errors.")
    S.append("")
    S.append(f"Linear LSQ condition number of the input matrix: {lin.cond:.1f}")
    S.append(f"Ridge penalty chosen on validation runs: {models['Ridge'].alpha:g}")
    S.append(f"Neural network: two hidden layers of {nn.width}, {nn.act}, L2 {nn.l2:g}, ensemble of {len(nn.members)} "
             f"seeds; early stopping at epochs {[i['best_epoch'] for i in nn.infos]}")
    S.append("  Single-network test RMSE across seeds (mean ± std): "
             + ", ".join(f"{ax} {seed_rmse[:, j].mean():.4f} ± {seed_rmse[:, j].std():.4f}" for j, ax in enumerate(AXES)))
    if "GP" in models:
        gpm = models["GP"]
        S.append(f"GP: trained on {gpm.n_train} rows; fitted kernel {gpm.m.kernel_}")
        S.append("  Fraction of test points inside ±2σ (ideal about 95%): "
                 + ", ".join(f"{a} {v:.1f}%" for a, v in zip(cov["axis"], cov["coverage_2sigma_%"])))
    S.append("")
    best_fz = mt["RMSE_Fz"].idxmin()
    def rel(new, old):
        p = 100 * (new - old) / old
        return f"{abs(p):.0f}% {'lower' if p < 0 else 'higher'} than"
    S.append(f"Lowest axial test RMSE: {best_fz}, {mt.loc[best_fz, 'RMSE_Fz']:.4f} N, "
             f"{rel(mt.loc[best_fz, 'RMSE_Fz'], PRIOR_RMSE['Fz'])} the prior sensor's 0.233 N training error")
    lin_fz = mt.loc["Linear LSQ", "RMSE_Fz"]
    S.append(f"Neural network axial RMSE {mt.loc['Neural network', 'RMSE_Fz']:.4f} N is "
             f"{rel(mt.loc['Neural network', 'RMSE_Fz'], lin_fz)} linear LSQ ({lin_fz:.4f} N)")
    if cv is not None:
        S.append("")
        S.append("Grouped 5-fold cross-validation, RMSE mean ± std (N) [report Table 6.3]")
        for n in [m for m in MODEL_ORDER if m in cv.index]:
            S.append(f"  {n:<16s}" + "  ".join(f"{ax} {cv.loc[n, ('RMSE_' + ax, 'mean')]:.4f} ± {cv.loc[n, ('RMSE_' + ax, 'std')]:.4f}" for ax in AXES))
    if lc is not None:
        S.append("")
        S.append("Learning curve, axial test RMSE (N) by training runs used [report Section 6.6]")
        piv = lc.groupby(["model", "n_runs"])["RMSE_Fz"].mean().unstack()
        S.append("  " + piv.round(4).to_string().replace("\n", "\n  "))
    if hang is not None:
        S.append("")
        S.append("Hanging-mass check: see metrics_hanging_mass.csv")
    S.append("")
    S.append("Figures (results/figures/): pred_vs_true, residual_vs_true, residual_vs_tilt, axial_rmse_vs_tilt,")
    S.append("  nn_training_curve, calibration_matrix, gp_uncertainty, learning_curve, raw_response")
    S.append(f"Run time {time.time() - t0:.0f} s")
    txt = "\n".join(S)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:   # utf-8 so σ, ° and ± save on Windows
        f.write(txt + "\n")
    log("\n" + txt)


if __name__ == "__main__":
    main()