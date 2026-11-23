#!/usr/bin/env python3
"""
live_force.py - shows the live 3-axis force from the FBG sensor.

Reads the fibre wavelengths from the si155, takes a zero reading (tare), then uses a
model from fit_models.py to show Fx, Fy and Fz as the sensor is pressed. With the GP
model it also shows an error band, and warns when a reading is outside the training data.

Usage:
  python live_force.py --sim                              # no hardware: replays test runs
  python live_force.py --sim --hysteresis 0.1             # replay with a leftover offset,
                                                          #   to try Re-zero
  python live_force.py --si155 10.0.0.55                  # live sensor
  python live_force.py --si155 10.0.0.55 --model "Neural network" --log press1.csv
  python live_force.py --models results/models.joblib     # use a different set of models

Keys:
  t  tare (2 s zero reading, keep the tip clear)
  z  re-zero (quick 0.5 s tare, for a leftover force after a press; same as the button)
  m  next model
  b  show / hide the GP band
  v  reset the 3D view (drag the 3D view to rotate it)
  p  pause / resume
  s  save a screenshot
  q  quit

The models work on wavelength shifts from the zero reading, so the display always
tares first. f1, f2 and f3 must be the same fibres as in the training data, so keep
FIBRE_MAP the same as in collect.py.

Needs numpy, pandas, matplotlib, scikit-learn and joblib, with fit_models.py in the
same folder.
"""

import argparse
import collections
import csv
import datetime as dt
import math
import os
import sys
import threading
import time

import numpy as np

# =============================== CONFIG ===============================
SI155_IP = "10.0.0.55"
SI155_CHANNELS = (1, 2, 3, 4)
SI155_POLL_HZ = 50           # readings asked for per second. The display needs ~12,
                             #   and asking much faster can stall the si155
SI155_READ_TIMEOUT_S = 2.0   # restart the reader if nothing arrives for this long
SI155_RECONNECT_EVERY = 0    # 0 = keep one connection open (normal). N = reconnect every
                             #   N reads (only if --check says to)
# fibre name -> (channel, peak number), the same as in collect.py
FIBRE_MAP = {
    "f1": (1, 0),
    "f2": (2, 0),
    "f3": (3, 0),
}
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(HERE, "results_prior", "models.joblib")      # paths are relative to this
DATASET_FOR_SIM = os.path.join(HERE, "dataset_prior", "dataset_all.csv")  # script, not the current folder
DEFAULT_MODEL = "GP"

DISPLAY_HZ = 12              # screen updates per second (about 12 is the most a redraw
                             #   allows; the si155 is still read faster in the background)
AVG_SAMPLES = 5              # readings averaged into each point shown
TARE_SECONDS = 2.0           # length of the tare at startup and on 't'
REZERO_SECONDS = 0.5         # length of the quick re-zero ('z' or the button)
REZERO_WARN_N = 0.25         # warn if re-zeroed while showing more than this
SMOOTHING = 0.35             # smoothing of the force (0 = off, 0.9 = heavy)
WINDOW_S = 10.0              # seconds shown on the scrolling traces
ZERO_OFFSET_ON_TARE = True   # make the display read exactly 0 N after a tare
VIEW_ELEV, VIEW_AZIM = 22, -55   # starting 3D view (drag to rotate, v resets)
ARROW_MODE = "push"          # "push":   the arrow points into the tip, like the force applied
                             # "vector": the arrow points out from the tip to (Fx, Fy, Fz)
                             # The numbers are the same either way (Fz > 0 means pressing in).
DISPLAY_LIMIT_N = 0.5        # axis range for the 3D view, traces and bar (N)
DISPLAY_FLOOR_N = -0.1       # lowest Fz shown
PROTRACTOR = True            # rig x-angle scale on the floor of the 3D view
PROTRACTOR_OFFSET_DEG = 0.0  # keep equal to ALPHA_OFFSET_DEG in build_dataset.py, so a run
                             #   at x angle X points at the X mark
FORCE_LIMIT_N = 0.5          # warn above this force (the 2026 runs reach about 0.45 N)
SIGMA_WARN_RATIO = 2.5       # warn when the GP's error bar is this many times its usual size
SIM_NOISE_PM = 2.0           # noise added in replay mode (pm)
# ======================================================================

AXES = ("Fx", "Fy", "Fz")
# Colours
COL = {"Fx": "#2a78d6", "Fy": "#eb6834", "Fz": "#1baf7a"}
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#1f1f1e", "#5f5e58", "#9a998f"
GRID = "#e6e5df"
WARN_COL = "#b25400"


# ============================== MODELS ================================
def load_models(path):
    """Load models.joblib from fit_models.py. The models were saved while fit_models.py
    was run directly, so their classes need pointing back at fit_models first."""
    import joblib
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import fit_models
    except ImportError:
        sys.exit("fit_models.py must be in the same folder as live_force.py.")
    main_mod = sys.modules["__main__"]
    for name in ("LinearLSQ", "RidgeModel", "QuadraticModel", "NNEnsemble", "GPModel"):
        setattr(main_mod, name, getattr(fit_models, name))
    if not os.path.exists(path):
        sys.exit(f"No models at {path}.\nTrain them first:  python fit_models.py --dataset dataset_prior "
                 "--out results_prior\n(or run run_prior_demo.py, which does every step).")
    saved = joblib.load(path)
    return saved["models"], saved["inputs"]


class Estimator:
    """Handles the tare, the shifts, the model prediction, smoothing and warnings."""

    def __init__(self, models, inputs, model_name):
        self.models, self.inputs = models, inputs
        self.names = [n for n in ("Linear LSQ", "Ridge", "Quadratic", "Neural network", "GP") if n in models]
        if model_name not in self.models:
            sys.exit(f"Model '{model_name}' not in {self.names}")
        self.model_name = model_name
        self.n_in = len(inputs)
        self.baseline = None
        self.offsets = {n: np.zeros(3) for n in self.names}
        self.smooth = None
        self.gp = models.get("GP")
        self.sigma_typ = None
        self.in_lo = self.in_hi = None
        if self.gp is not None:
            Xtr = self.gp.sx.inverse_transform(self.gp.m.X_train_)
            self.in_lo, self.in_hi = Xtr.min(axis=0), Xtr.max(axis=0)
            _, sd = self.gp.predict(Xtr[:: max(1, len(Xtr) // 200)], return_std=True)
            self.sigma_typ = np.median(sd, axis=0)
        if ZERO_OFFSET_ON_TARE:
            z = np.zeros((1, self.n_in))
            for n in self.names:
                self.offsets[n] = self.models[n].predict(z)[0]

    def next_model(self):
        i = self.names.index(self.model_name)
        self.model_name = self.names[(i + 1) % len(self.names)]
        self.smooth = None

    def tare(self, lam_samples):
        self.baseline = np.mean(np.asarray(lam_samples), axis=0)
        self.smooth = None

    def shifts_pm(self, lam):
        return (np.asarray(lam) - self.baseline) * 1000.0

    def predict(self, dlam):
        X = np.asarray(dlam, float)[: self.n_in].reshape(1, -1)
        F = self.models[self.model_name].predict(X)[0] - self.offsets[self.model_name]
        sd = None
        if self.gp is not None:
            _, s = self.gp.predict(X, return_std=True)
            sd = s[0]
        if SMOOTHING > 0 and self.smooth is not None:
            F = SMOOTHING * self.smooth + (1 - SMOOTHING) * F
        self.smooth = F
        return F, sd

    def confidence(self, dlam, sd, F):
        """Returns (ok, message)."""
        msgs = []
        if self.in_lo is not None:
            x = np.asarray(dlam)[: self.n_in]
            span = self.in_hi - self.in_lo
            if np.any(x < self.in_lo - 0.05 * span) or np.any(x > self.in_hi + 0.05 * span):
                msgs.append("shifts outside the training range")
        if sd is not None and self.sigma_typ is not None:
            if np.any(sd > SIGMA_WARN_RATIO * self.sigma_typ):
                msgs.append("GP uncertainty high")
        if np.linalg.norm(F) > FORCE_LIMIT_N:
            msgs.append("force beyond calibrated range")
        return (not msgs), "; ".join(msgs)


# ============================== SOURCES ===============================
class Si155Client:
    """Simple client for talking to the si155 over the network. Luna's own Python library
    kept opening new connections without closing the old ones, so the si155 stopped
    answering after about 10 reads. This one keeps a single connection open, gives up on
    a read that takes too long, and closes cleanly on any error. collect.py uses the same
    client."""

    COMMAND_PORT = 51971

    def __init__(self, ip, timeout=1.5):
        import socket
        self.sock = socket.create_connection((ip, self.COMMAND_PORT), timeout=timeout)
        self.sock.settimeout(timeout)
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("the si155 closed the connection")
            buf += chunk
        return bytes(buf)

    def command(self, cmd, arg=""):
        import struct
        c, a = cmd.encode("ascii"), arg.encode("ascii")
        self.sock.sendall(struct.pack("<BBHI", 0, 0, len(c), len(a)) + c + a)
        status, _rtype, mlen, clen = struct.unpack("<BBHI", self._recv(8))
        msg = self._recv(mlen) if mlen else b""
        if status != 0:
            raise RuntimeError(f"si155 refused {cmd}: {msg.decode('ascii', 'replace')}")
        return self._recv(clen) if clen else b""

    def peaks(self):
        """Peak wavelengths (nm) for each channel."""
        import struct
        raw = self.command("#GetPeaks")
        length = struct.unpack("<HHIQII", raw[:24])[0]
        counts = np.frombuffer(raw[24:length], dtype="<i2")
        data = np.frombuffer(raw[length:], dtype="<f8")
        out, start = {}, 0
        for ch, n in enumerate(counts, start=1):
            out[ch] = data[start:start + int(n)].tolist()
            start += int(n)
        return out

    def serial_number(self):
        return self.command("#GetSerialNumber").decode("ascii", "replace").strip("\x00 ")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def _si155_worker(ip, fibre_map, reconnect_every, poll_hz, q, stop):
    """Reads the si155 in a separate process, so a stuck read can't freeze the display.
    If it gets stuck, the display ends this process and starts a new one."""
    cli, n_conn, backoff = None, 0, 0.3
    period = 1.0 / poll_hz
    t_next = time.time()
    while not stop.is_set():
        now = time.time()
        if now < t_next:
            time.sleep(t_next - now)
        t_next = max(t_next + period, time.time() - period)
        try:
            if cli is None:
                cli, n_conn = Si155Client(ip, timeout=SI155_READ_TIMEOUT_S * 0.75), 0
                q.put(("info", time.time(), cli.serial_number()))
            peaks = cli.peaks()
            raw, vals, missing = {}, [], False
            for name, (ch, idx) in fibre_map.items():
                lst = peaks.get(ch, [])
                raw[ch] = lst
                if idx < len(lst):
                    vals.append(lst[idx])
                else:
                    missing = True
            n_conn += 1
            q.put(("missing" if missing else "ok", time.time(), vals, raw, n_conn))
            backoff = 0.3
            if reconnect_every and n_conn >= reconnect_every:
                cli.close()                     # planned reconnect, not an error
                cli = None
        except Exception as e:
            q.put(("err", time.time(), f"{type(e).__name__}: {e}", n_conn))
            if cli is not None:
                cli.close()
            cli = None
            time.sleep(backoff)
            backoff = min(backoff * 2, 3.0)
    if cli is not None:
        cli.close()


class SI155Source:
    """Gets readings from _si155_worker and keeps the latest ones for the display.
    If the worker goes quiet for SI155_READ_TIMEOUT_S, it is restarted."""

    STALE_S = 3.0          # show "no data" if the newest reading is older than this
    STARTUP_S = 10.0       # time allowed for a new worker to connect

    def __init__(self, ip, debug=False, reconnect_every=None, poll_hz=None, quiet=False):
        import logging
        import multiprocessing as mp
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)
        self.ctx = mp.get_context("spawn")     # what Windows uses, so the same on every system
        self.ip, self.debug, self.quiet = ip, debug, quiet
        self.reconnect_every = SI155_RECONNECT_EVERY if reconnect_every is None else reconnect_every
        # reconnecting every read is slower, so ask less often
        self.poll_hz = poll_hz or (min(SI155_POLL_HZ, 20) if self.reconnect_every == 1 else SI155_POLL_HZ)
        self.buf = collections.deque(maxlen=2000)
        self.lock = threading.Lock()
        self.error = ""
        self.n_read = self.n_reconnect = self.n_timeout = 0
        self.truth = None
        self.last_ok = 0.0
        self.last_peaks = None
        self.serial = "?"
        self.proc = self.q = self.stop_ev = None
        self._first = threading.Event()
        self._first_err = None
        self._closing = threading.Event()
        self._start_worker()
        self._drainer = threading.Thread(target=self._drain, daemon=True)
        self._drainer.start()
        if not self._first.wait(self.STARTUP_S + 5):
            self.close()
            sys.exit(f"No reply from the si155 at {ip}. Check the Ethernet cable and IP address, "
                     "close ENLIGHT or any other program using it, and power-cycle the si155.")
        if self._first_err:
            self.close()
            sys.exit(f"Could not read from the si155 at {ip}: {self._first_err}")

    # ---------------------------------------------------------- worker process
    def _start_worker(self):
        self.q, self.stop_ev = self.ctx.Queue(), self.ctx.Event()
        self.proc = self.ctx.Process(target=_si155_worker, daemon=True,
                                     args=(self.ip, dict(FIBRE_MAP), self.reconnect_every,
                                           self.poll_hz, self.q, self.stop_ev))
        self.proc.start()
        self.t_worker = time.time()

    def _end_worker(self):
        try:
            self.stop_ev.set()
            self.proc.join(0.3)
            if self.proc.is_alive():
                self.proc.terminate()          # stuck read: end the process to free its connection
                self.proc.join(1.0)
        except Exception:
            pass

    def _log(self, msg):
        n = self.n_reconnect
        if not self.quiet and (n <= 5 or n % 20 == 0):   # don't flood the console
            print(msg + ("" if n <= 5 else "  (printing every 20th from now on)"), flush=True)

    def _drain(self):
        import queue as _queue
        while not self._closing.is_set():
            try:
                msg = self.q.get(timeout=0.2)
            except (_queue.Empty, EOFError, OSError):
                msg = None
            if msg:
                kind = msg[0]
                if kind == "info":
                    self.serial = msg[2]
                elif kind == "fatal":
                    self._first_err = msg[2]
                    self._first.set()
                elif kind == "ok":
                    _, t, vals, raw, _n = msg
                    with self.lock:
                        self.buf.append((t, vals))
                        self.n_read += 1
                    self.last_ok, self.last_peaks, self.error = t, raw, ""
                    self._first.set()
                elif kind == "missing":
                    self.last_peaks = msg[3]
                    self.error = "a fibre's peak is missing (check FIBRE_MAP or the fibre)"
                    if not self._first.is_set():
                        self._first_err = ("connected, but FIBRE_MAP points at a peak that isn't there. "
                                           f"Peaks per channel: { {c: len(l) for c, l in msg[3].items()} }")
                        self._first.set()
                elif kind == "err":
                    self.n_reconnect += 1
                    self.error = "reconnecting to si155 (" + msg[2].split(":")[0] + ")"
                    self._log(f"si155: {msg[2]} after {msg[3]} reads on this connection. "
                              f"Reconnecting ({self.n_reconnect}).")
            # no reply from the si155: restart the worker
            if self.last_ok >= self.t_worker:   # time since this worker's last reading
                ref = self.last_ok
            else:                               # still starting up
                ref = self.t_worker + self.STARTUP_S - SI155_READ_TIMEOUT_S
            if not self._closing.is_set() and time.time() - ref > SI155_READ_TIMEOUT_S:
                self.n_reconnect += 1
                self.n_timeout += 1
                self.error = "si155 stopped answering, restarting the reader"
                self._log(f"si155: no reading for {time.time() - max(self.last_ok, self.t_worker):.1f} s. "
                          f"Restarting the reader ({self.n_reconnect}).")
                self._end_worker()
                self._start_worker()

    # ---------------------------------------------------------- used by the display
    def latest(self, n):
        with self.lock:
            items = list(self.buf)[-n:]
        if not items:
            return None
        if time.time() - items[-1][0] > self.STALE_S:
            return None                        # don't show an old reading as live
        return np.mean([v for _, v in items], axis=0)

    def collect(self, seconds):
        t0 = time.time()
        time.sleep(seconds)
        with self.lock:
            items = [v for t, v in self.buf if t >= t0]
        return items

    def close(self):
        self._closing.set()
        self._end_worker()


def check_si155(ip, seconds=5.0):
    """--check: test the si155 link without the display. Tries one kept-open connection,
    then a new connection for each read, and says which works."""
    print(f"Connecting to the si155 at {ip} ...")
    results = {}
    tests = [("A", "one connection kept open", 0, SI155_POLL_HZ),
             ("B", "fresh connection for every read", 1, 20)]
    for key, label, rec, hz in tests:
        print(f"\nTest {key}: {label}, asking {hz} times a second, {seconds:.0f} s")
        src = SI155Source(ip, reconnect_every=rec, poll_hz=hz, quiet=True)
        if key == "A":
            print(f"  si155 serial number: {src.serial}  (talking to it directly, not through hyperion.py)")
            raw = src.last_peaks or {}
            print("  Peaks per channel (nm): " + "   ".join(
                f"ch{ch}: " + (", ".join(f"{v:.4f}" for v in raw[ch]) or "none") for ch in sorted(raw)))
            print("  FIBRE_MAP picks: " + ", ".join(f"{k} = ch{c} peak {i}" for k, (c, i) in FIBRE_MAP.items()))
        n0, t0 = src.n_read, time.time()
        last_n = n0
        while time.time() - t0 < seconds:
            time.sleep(1.0)
            got = src.n_read - last_n
            last_n = src.n_read
            v = src.latest(1)
            txt = "no data" if v is None else ", ".join(f"{k} {x:.4f}" for k, x in zip(FIBRE_MAP, v))
            print(f"  {time.time() - t0:4.1f} s  {got:3d} reads   {txt}   {src.error}")
        rate = (src.n_read - n0) / (time.time() - t0)
        print(f"  -> {rate:.0f} reads/s, {src.n_reconnect} problems "
              f"({src.n_timeout} reads never answered, reader restarted)")
        results[key] = (rate, src.n_reconnect, hz)
        src.close()
        time.sleep(0.5)

    (ra, fa, ha), (rb, fb, hb) = results["A"], results["B"]
    print()
    if fa == 0 and ra > 0.8 * ha:
        print("RESULT: OK. Keep SI155_RECONNECT_EVERY = 0. The display is ready.")
    elif fb == 0 and rb > 0.5 * hb:
        print("RESULT: a kept-open connection stalls on this si155, a fresh one per read is fine.")
        print("  Use:  python live_force.py --si155 %s --reconnect-every 1" % ip)
        print("  (or set SI155_RECONNECT_EVERY = 1 at the top of live_force.py)")
    else:
        print("RESULT: both ways stall. Close ENLIGHT and any other script using the si155, power-cycle")
        print("  the si155, and check Luna's hyperion.py port fix (see the README's si155 section).")
        print("  Then run --check again with --debug and send me the output.")


class ReplaySource:
    """No hardware: replays test runs as if someone were pressing the sensor (press, hold,
    release, rest) and shows the true force alongside."""

    BASE_NM = np.array([1550.684, 1550.519, 1550.651])

    def __init__(self, dataset, inputs, split="test", seed=0, hysteresis=0.0):
        import pandas as pd
        if not os.path.exists(dataset):
            sys.exit(f"--sim needs {dataset} (run prior_to_dataset.py first).")
        d = pd.read_csv(dataset)
        d = d[d["split"] == split]
        rng = np.random.default_rng(seed)
        runs = list(d["run_id"].unique())
        rng.shuffle(runs)
        self.inputs, self.rng = inputs, rng
        self.segments = []      # (shifts, forces, label) for each run
        for r in runs:
            g = d[d["run_id"] == r].sort_values("F")
            S = np.vstack([np.zeros(len(inputs)), g[inputs].to_numpy(float)])
            Fv = np.vstack([np.zeros(3), g[list(AXES)].to_numpy(float)])
            lab = f"tilt {g['beta_deg'].iloc[0]:g}°, rotation {g['alpha_deg'].iloc[0]:g}°"
            self.segments.append((S, Fv, lab))
        self.hysteresis = hysteresis   # share of each press's shift left behind afterwards
        self.t0 = time.time()
        self.rest, self.up, self.hold = 1.5, 5.0, 1.5
        self.cycle = self.rest + 2 * self.up + self.hold
        self.error = ""
        self.truth = np.zeros(3)
        self.label = ""
        self.n_read = 0

    def _state(self, t):
        k = int(t // self.cycle) % len(self.segments)
        u = t % self.cycle
        S, Fv, lab = self.segments[k]
        n = len(S) - 1
        if u < self.rest:
            pos = 0.0
        elif u < self.rest + self.up:
            pos = (u - self.rest) / self.up * n
        elif u < self.rest + self.up + self.hold:
            pos = n
        else:
            pos = max(0.0, n - (u - self.rest - self.up - self.hold) / self.up * n)
        i = min(int(pos), n - 1)
        w = pos - i
        s = (1 - w) * S[i] + w * S[i + 1]
        f = (1 - w) * Fv[i] + w * Fv[i + 1]
        if self.hysteresis:
            # each finished press leaves some shift behind, like silicone that hasn't
            # relaxed, so the display drifts while the true force goes back to zero
            done = int(t // self.cycle)
            N = len(self.segments)
            s = s + self.hysteresis * sum(self.segments[j % N][0][-1] for j in range(done))
        return s, f, lab

    def latest(self, n):
        s, f, lab = self._state(time.time() - self.t0)
        self.truth, self.label = f, lab
        self.n_read += n
        noise = self.rng.normal(0, SIM_NOISE_PM / math.sqrt(max(n, 1)), size=len(s))
        lam = self.BASE_NM[: len(s)] + (s + noise) / 1000.0
        return lam

    def collect(self, seconds):
        # the replay starts at rest, so the tare sees zero load
        k = max(1, int(seconds * DISPLAY_HZ))
        out = [self.latest(1) for _ in range(k)]
        return out

    def close(self):
        pass


# ============================== DISPLAY ===============================
class LiveDisplay:
    def __init__(self, src, est, log_path=None, show_band=True, headless=False):
        import matplotlib
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.src, self.est = src, est
        self.show_band = show_band
        self.paused = False
        self.sim = isinstance(src, ReplaySource)
        n = int(WINDOW_S * DISPLAY_HZ)
        self.t = collections.deque(maxlen=n)
        self.F = {a: collections.deque(maxlen=n) for a in AXES}
        self.S = {a: collections.deque(maxlen=n) for a in AXES}
        self.T = {a: collections.deque(maxlen=n) for a in AXES}
        self.t_start = time.time()
        self.n_zero = 0               # re-zeros since startup
        self.zero_marks = []          # (time, lines) for each re-zero mark
        self.flash_msg, self.flash_until, self.flash_warn = "", 0.0, False
        self.tared_at = None
        self.log_fh = self.log_w = None
        if log_path:
            self.log_fh = open(log_path, "w", newline="", encoding="utf-8")
            self.log_w = csv.writer(self.log_fh)
            self.log_w.writerow(["timestamp", "model"] + [f"{k}_nm" for k in FIBRE_MAP]
                                + [f"dlam_{k}_pm" for k in FIBRE_MAP]
                                + ["Fx_N", "Fy_N", "Fz_N", "sigma_Fx_N", "sigma_Fy_N", "sigma_Fz_N"]
                                + (["true_Fx_N", "true_Fy_N", "true_Fz_N"] if self.sim else [])
                                + ["zero_count"])
        self._build()

    # ---------------------------------------------------------- layout
    def _build(self):
        plt = self.plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                             "xtick.color": INK2, "ytick.color": INK2})
        self.fig = plt.figure(figsize=(13.5, 7.6), facecolor=SURFACE)
        try:
            self.fig.canvas.manager.set_window_title("FBG sensor: live force")
        except Exception:
            pass
        outer = self.fig.add_gridspec(1, 3, width_ratios=[1.45, 2.55, 0.55], wspace=0.22,
                                      left=0.055, right=0.985, top=0.88, bottom=0.075)
        g_tr = outer[0].subgridspec(3, 1, hspace=0.35)                       # traces
        g_mid = outer[1].subgridspec(2, 1, height_ratios=[3.3, 1], hspace=0.02)  # 3D, readout
        g_rt = outer[2].subgridspec(2, 1, height_ratios=[3.3, 1], hspace=0.02)   # axial bar
        self.ax_tr = {}
        for i, a in enumerate(AXES):
            ax = self.fig.add_subplot(g_tr[i, 0], facecolor=SURFACE,
                                      sharex=self.ax_tr["Fx"] if i else None)
            ax.set_xlim(-WINDOW_S, 0)
            ax.set_ylim(-DISPLAY_LIMIT_N, DISPLAY_LIMIT_N) if a != "Fz" else ax.set_ylim(DISPLAY_FLOOR_N, DISPLAY_LIMIT_N)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.axhline(0, color=INK3, lw=0.8)
            ax.set_ylabel(f"{a} (N)", color=INK)
            ax.text(0.005, 0.95, a, transform=ax.transAxes, va="top", ha="left",
                    fontsize=12, fontweight="bold", color=INK)
            self.ax_tr[a] = ax
            if i < 2:
                plt.setp(ax.get_xticklabels(), visible=False)
        self.ax_tr["Fz"].set_xlabel("Time (s)")
        self.line = {a: self.ax_tr[a].plot([], [], color=COL[a], lw=2, label="Estimate")[0] for a in AXES}
        self.true_line = {a: self.ax_tr[a].plot([], [], color=INK, lw=1.2, ls="--", label="True (replay)")[0]
                          for a in AXES}
        self.band = {a: None for a in AXES}
        handles = [self.line["Fx"]]
        labels = ["Estimate"]
        if self.est.gp is not None:
            from matplotlib.patches import Patch
            handles.append(Patch(color=INK3, alpha=0.3))
            labels.append("GP ±2σ")
        if self.sim:
            handles.append(self.true_line["Fx"])
            labels.append("True force (replay)")
        self.fig.legend(handles, labels, loc="upper left", fontsize=8.5, frameon=False, ncol=3,
                        bbox_to_anchor=(0.05, 0.935))

        # 3D force vector
        ax = self.fig.add_subplot(g_mid[0], projection="3d", facecolor=SURFACE)
        L, z0 = DISPLAY_LIMIT_N, DISPLAY_FLOOR_N
        self.z_floor = z0
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(z0, L)
        ax.set_box_aspect((2 * L, 2 * L, L - z0), zoom=1.32)  # same scale on every axis
        ax.set_xlabel("Fx", labelpad=-2); ax.set_ylabel("Fy", labelpad=-2)
        ax.set_zlabel("")   # vertical = sensor axis; a label here would overlap the bar
        ax.tick_params(labelsize=7.5, pad=-2)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0.97, 0.97, 0.96, 1.0))
            a._axinfo["grid"]["color"] = GRID
            a._axinfo["grid"]["linewidth"] = 0.7
        th = np.linspace(0, 2 * np.pi, 120)
        for r in (L / 2, L):                                 # magnitude rings on the floor
            ax.plot(r * np.cos(th), r * np.sin(th), z0, color=INK3, lw=0.7, alpha=0.6)
            ax.text(r * 0.707, -r * 0.707, z0, f"{r:g} N", fontsize=7, color=INK3)
        if PROTRACTOR:                                       # rig x angle round the outer ring
            for xa in range(0, 360, 15):
                a = math.radians(xa + PROTRACTOR_OFFSET_DEG)
                ux, uy = math.sin(a), math.cos(a)            # direction of a run at x = xa
                major = xa % 30 == 0
                r_in = L * (0.90 if major else 0.95)
                ax.plot([r_in * ux, L * ux], [r_in * uy, L * uy], [z0, z0], color=INK2 if major else INK3,
                        lw=1.0 if major else 0.7)
                if major:
                    ax.text(0.80 * L * ux, 0.80 * L * uy, z0, f"{xa}°", fontsize=7.5, color=INK2,
                            ha="center", va="center")
            a0 = math.radians(PROTRACTOR_OFFSET_DEG)
            ax.plot([0, L * math.sin(a0)], [0, L * math.cos(a0)], [z0, z0], color=INK3, lw=0.7,
                    ls=(0, (2, 3)))                          # faint spoke to the x = 0 mark
        ax.plot([0, 0], [0, 0], [0, L], color=GRID, lw=0.9)            # sensor axis above the tip
        ax.plot([0, 0], [0, 0], [z0, 0], color=INK3, lw=6, solid_capstyle="butt")  # sensor body
        ax.scatter([0], [0], [0], color=INK, s=18, depthshade=False)
        ax.set_title("Force vector (N)", color=INK, fontsize=11, loc="left")
        ax.text2D(1.0, 1.035, "drag to rotate · v resets", transform=ax.transAxes, ha="right",
                  fontsize=7.5, color=INK3)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        self.ax3d = ax
        self.quiver = None
        self.shadow = ax.plot([], [], [], color=INK3, lw=1.5, ls="--")[0]   # lateral part on the floor
        self.drop = ax.plot([], [], [], color=INK3, lw=0.9, ls=":")[0]      # tip down to its shadow
        if self.sim:
            self.true_vec = ax.plot([], [], [], color=INK2, lw=1.2, ls="--")[0]
            self.true_tip = ax.plot([], [], [], "o", mfc="none", mec=INK2, ms=8, mew=1.3)[0]

        # axial force bar
        ax = self.fig.add_subplot(g_rt[0], facecolor=SURFACE)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(DISPLAY_FLOOR_N, DISPLAY_LIMIT_N)
        ax.set_xticks([])
        ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.axhline(0, color=INK3, lw=0.8)
        ax.set_title("Axial force (N)", color=INK, fontsize=11, loc="left")
        self.bar = ax.bar([0], [0], width=0.5, color=COL["Fz"])[0]
        self.bar_err = ax.plot([0, 0], [0, 0], color=INK, lw=2, solid_capstyle="butt")[0]
        self.bar_true = ax.plot([-0.32, 0.32], [np.nan, np.nan], color=INK, lw=1.5, ls="--")[0] if self.sim else None
        self.ax_bar = ax

        # readout
        ax = self.fig.add_subplot(g_mid[1], facecolor=SURFACE)
        ax.axis("off")
        self.txt_vals = ax.text(0.0, 0.84, "", va="top", ha="left", family="DejaVu Sans Mono", fontsize=12.5,
                                color=INK, transform=ax.transAxes)
        self.txt_info = ax.text(0.0, 0.34, "", va="top", ha="left", fontsize=9, color=INK2,
                                transform=ax.transAxes)
        self.title = self.fig.text(0.06, 0.965, "", fontsize=14, fontweight="bold", color=INK, va="center")
        self.status = self.fig.text(0.985, 0.965, "", fontsize=11, ha="right", va="center", color=INK2)
        self.fig.text(0.985, 0.015, "z re-zero   t tare   m model   b band   v reset 3D view   p pause   "
                      "s screenshot   q quit", ha="right", fontsize=8.5, color=INK3)
        self.fig.canvas.mpl_connect("key_press_event", self._key)

        # Re-zero button, under the axial bar
        from matplotlib.widgets import Button
        cell = g_rt[1].get_position(self.fig)
        bx = self.fig.add_axes([cell.x0 + 0.08 * cell.width, cell.y0 + 0.42 * cell.height,
                                0.84 * cell.width, 0.36 * cell.height])
        self.btn = Button(bx, "Re-zero", color="#ecebe6", hovercolor="#dcdbd4")
        self.btn.label.set_fontsize(12); self.btn.label.set_fontweight("bold"); self.btn.label.set_color(INK)
        for sp in bx.spines.values():
            sp.set_edgecolor(INK3)
        self.btn.on_clicked(lambda _ev: self.rezero())
        self.txt_zero = self.fig.text(cell.x0 + 0.5 * cell.width, cell.y0 + 0.30 * cell.height, "",
                                      ha="center", va="top", fontsize=8, color=INK2)

    # ---------------------------------------------------------- actions
    def tare(self, seconds=TARE_SECONDS):
        self.status.set_text("Taring... keep the tip clear")
        self.status.set_color(WARN_COL)
        try:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)
        except Exception:
            pass
        samples = self.src.collect(seconds)
        if not samples:
            print("Tare failed: no readings from the si155 yet. Trying again.")
            self.status.set_text("⚠ Waiting for si155 data to tare")
            return
        self.est.tare(samples)
        self.tared_at = time.time()
        print(f"Tared on {len(samples)} readings: " + ", ".join(
            f"{k} {v:.4f} nm" for k, v in zip(FIBRE_MAP, self.est.baseline)))

    def rezero(self, seconds=REZERO_SECONDS):
        """Quick tare for when the tip is unloaded but the silicone hasn't fully relaxed,
        so the display still shows a force."""
        shown = float(np.linalg.norm(self.est.smooth)) if self.est.smooth is not None else 0.0
        self.tare(seconds)
        if self.est.baseline is None:
            return
        self.n_zero += 1
        t_mark = time.time() - self.t_start
        lines = [self.ax_tr[a].axvline(0, color=INK2, lw=1, ls=":") for a in AXES]
        self.zero_marks.append((t_mark, lines))
        self.flash_warn = shown > REZERO_WARN_N
        self.flash_msg = (f"⚠ Re-zeroed while showing {shown:.2f} N. Was the tip clear?" if self.flash_warn
                          else f"Re-zeroed (cleared {shown:.2f} N)")
        self.flash_until = time.time() + 3.0
        print(self.flash_msg.replace("⚠ ", "WARNING: "))
    def _key(self, ev):
        k = (ev.key or "").lower()
        if k == "t":
            self.tare()
        elif k == "z":
            self.rezero()
        elif k == "m":
            self.est.next_model()
        elif k == "b":
            self.show_band = not self.show_band
        elif k == "p":
            self.paused = not self.paused
        elif k == "s":
            fn = f"live_{dt.datetime.now():%Y%m%d_%H%M%S}.png"
            self.fig.savefig(fn, dpi=150, facecolor=SURFACE)
            print(f"Saved {fn}")
        elif k == "v":
            self.ax3d.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        elif k == "q":
            self.plt.close(self.fig)

    # ---------------------------------------------------------- one frame
    def step(self):
        if self.paused:
            return
        if self.est.baseline is None:          # tare failed (no data yet), keep trying
            self.tare()
            return
        lam = self.src.latest(AVG_SAMPLES)
        if lam is None:
            err = getattr(self.src, "error", "") or "waiting for data"
            n = getattr(self.src, "n_reconnect", 0)
            self.status.set_text(f"⚠ No data: {err}" + (f" (reconnects {n})" if n else ""))
            self.status.set_color(WARN_COL)
            return
        dlam = self.est.shifts_pm(lam)
        F, sd = self.est.predict(dlam)
        now = time.time() - self.t_start
        self.t.append(now)
        truth = getattr(self.src, "truth", None)
        for j, a in enumerate(AXES):
            self.F[a].append(F[j])
            self.S[a].append(sd[j] if sd is not None else np.nan)
            self.T[a].append(truth[j] if (self.sim and truth is not None) else np.nan)
        ok, msg = self.est.confidence(dlam, sd, F)
        self._draw(F, sd, dlam, ok, msg, truth)
        if self.log_w:
            row = [dt.datetime.now().isoformat(timespec="milliseconds"), self.est.model_name]
            row += [f"{v:.5f}" for v in lam] + [f"{v:.2f}" for v in dlam] + [f"{v:.5f}" for v in F]
            row += [f"{v:.5f}" for v in sd] if sd is not None else ["", "", ""]
            if self.sim:
                row += [f"{v:.5f}" for v in truth]
            row.append(self.n_zero)
            self.log_w.writerow(row)

    def _draw(self, F, sd, dlam, ok, msg, truth):
        t = np.asarray(self.t) - self.t[-1]
        t_last = self.t[-1]
        keep = []
        for t_mark, lines in self.zero_marks:      # dotted line on the traces at each re-zero
            x = t_mark - t_last
            if x < -WINDOW_S:
                for ln in lines:
                    ln.remove()
            else:
                for ln in lines:
                    ln.set_xdata([x, x])
                keep.append((t_mark, lines))
        self.zero_marks = keep
        if self.tared_at is not None:
            ago = time.time() - self.tared_at
            self.txt_zero.set_text(f"zeroed {ago:.0f} s ago" + (f"\n{self.n_zero} re-zero"
                                   + ("s" if self.n_zero != 1 else "") if self.n_zero else ""))
        for a in AXES:
            y = np.asarray(self.F[a])
            self.line[a].set_data(t, y)
            if self.sim:
                self.true_line[a].set_data(t, np.asarray(self.T[a]))
            if self.band[a] is not None:
                self.band[a].remove()
                self.band[a] = None
            s = np.asarray(self.S[a])
            if self.show_band and self.est.gp is not None and np.isfinite(s).any():
                self.band[a] = self.ax_tr[a].fill_between(t, y - 2 * s, y + 2 * s, color=INK3, alpha=0.22, lw=0)

        # 3D arrow
        fx, fy, fz = F
        lat = math.hypot(fx, fy)
        mag = float(np.linalg.norm(F))
        push = ARROW_MODE == "push"
        if self.quiver is not None:
            self.quiver.remove()
            self.quiver = None
        if mag > 0.005:
            # fixed-size arrowhead, so small forces still show one
            ratio = min(0.45, 0.1 * DISPLAY_LIMIT_N / mag)
            if push:   # tail out at (Fx, Fy, Fz), head on the tip
                self.quiver = self.ax3d.quiver(fx, fy, fz, -fx, -fy, -fz, color=INK, linewidth=2.6,
                                               arrow_length_ratio=ratio)
            else:      # tail on the tip, head out at (Fx, Fy, Fz)
                self.quiver = self.ax3d.quiver(0, 0, 0, fx, fy, fz, color=INK, linewidth=2.6,
                                               arrow_length_ratio=ratio)
        # the shadow and drop line are the same for both arrow modes
        z0 = self.z_floor
        self.shadow.set_data_3d([0, fx], [0, fy], [z0, z0])
        self.drop.set_data_3d([fx, fx], [fy, fy], [fz, z0])
        if self.sim and truth is not None:
            tx, ty, tz = truth
            self.true_vec.set_data_3d([0, tx], [0, ty], [0, tz])
            self.true_tip.set_data_3d([tx], [ty], [tz])

        # axial bar
        self.bar.set_height(F[2])
        if sd is not None and self.show_band:
            self.bar_err.set_data([0, 0], [F[2] - 2 * sd[2], F[2] + 2 * sd[2]])
        else:
            self.bar_err.set_data([0, 0], [np.nan, np.nan])
        if self.sim and truth is not None:
            self.bar_true.set_data([-0.32, 0.32], [truth[2], truth[2]])

        # text
        def pm(j):
            return f"±{2 * sd[j]:.3f}" if sd is not None else ""
        # direction as a rig x angle (matches the protractor)
        ang = ((math.degrees(math.atan2(fx, fy)) - PROTRACTOR_OFFSET_DEG) % 360
               if lat > 0.02 else float("nan"))
        tilt = math.degrees(math.atan2(lat, fz)) if mag > 0.02 else float("nan")
        dname = "x angle"
        lines = ["   ".join(f"{a} {F[j]:+6.3f}{pm(j)}" for j, a in enumerate(AXES)) + "  N",
                 f"|F| {mag:5.3f} N   tilt from axis {'  -' if math.isnan(tilt) else f'{tilt:3.0f}°'}"
                 f"   {dname} {'   -' if math.isnan(ang) else f'{ang:4.0f}°'}"]
        self.txt_vals.set_text("\n".join(lines))
        info = (f"Shifts: " + "  ".join(f"{k} {v:+7.1f} pm" for k, v in zip(FIBRE_MAP, dlam)))
        if self.sim:
            info += f"\nReplaying held-out test run: {self.src.label}"
        if sd is not None:
            info += "\n± is the GP's 2σ (shown for every model).   Angles: tilt from the sensor axis; x angle as on the rig (protractor on the 3D floor)"
        self.txt_info.set_text(info)
        src = "REPLAY (no hardware)" if self.sim else f"si155 {self.src.ip}"
        self.title.set_text(f"Live force   ·   {self.est.model_name}   ·   {src}")
        err = getattr(self.src, "error", "")
        if time.time() < self.flash_until:
            self.status.set_text(self.flash_msg)
            self.status.set_color(WARN_COL if self.flash_warn else INK2)
        elif err:
            self.status.set_text(f"⚠ {err}"); self.status.set_color(WARN_COL)
        elif not ok:
            self.status.set_text(f"⚠ Low confidence: {msg}"); self.status.set_color(WARN_COL)
        else:
            n = getattr(self.src, "n_reconnect", 0)
            self.status.set_text("✓ Within calibrated range" + (f"   ·   si155 reconnects: {n}" if n else ""))
            self.status.set_color(INK2)

    # ---------------------------------------------------------- loops
    def run(self):
        from matplotlib.animation import FuncAnimation
        self.tare()
        self.anim = FuncAnimation(self.fig, lambda _: self.step(), interval=1000 / DISPLAY_HZ,
                                  cache_frame_data=False)
        self.plt.show()
        self.close()

    def run_headless(self, frames, save):
        self.tare()
        for _ in range(frames):
            t_next = time.time() + 1 / DISPLAY_HZ
            self.step()
            time.sleep(max(0.0, t_next - time.time()))
        self.fig.savefig(save, dpi=130, facecolor=SURFACE)
        print(f"Saved {save}")
        self.close()

    def close(self):
        self.src.close()
        if self.log_fh:
            self.log_fh.close()


def main():
    global DISPLAY_HZ, SMOOTHING
    ap = argparse.ArgumentParser(description="Live 3-axis force display for the FBG sensor")
    ap.add_argument("--si155", default=None, help=f"interrogator IP (e.g. {SI155_IP})")
    ap.add_argument("--sim", action="store_true", help="no hardware: replay held-out test runs")
    ap.add_argument("--models", default=MODELS_PATH, help="models.joblib from fit_models.py")
    ap.add_argument("--dataset", default=DATASET_FOR_SIM, help="dataset_all.csv used by --sim")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help='"GP", "Neural network", "Linear LSQ", "Ridge" or "Quadratic"')
    ap.add_argument("--log", default=None, help="write every displayed reading to this CSV")
    ap.add_argument("--hz", type=float, default=DISPLAY_HZ)
    ap.add_argument("--smoothing", type=float, default=SMOOTHING)
    ap.add_argument("--no-band", action="store_true", help="start with the GP band hidden")
    ap.add_argument("--frames", type=int, default=0, help="headless test: run N frames then save --save")
    ap.add_argument("--save", default="live_test.png")
    ap.add_argument("--seed", type=int, default=0, help="--sim: order the test runs are replayed in")
    ap.add_argument("--check", action="store_true",
                    help="with --si155: test the connection and FIBRE_MAP for 5 s, no display")
    ap.add_argument("--debug", action="store_true", help="print full errors from the si155 connection")
    ap.add_argument("--reconnect-every", type=int, default=None,
                    help="si155: open a fresh connection every N reads (1 = every read); "
                         "default SI155_RECONNECT_EVERY")
    ap.add_argument("--hysteresis", type=float, default=0.0,
                    help="--sim: fraction of each press's peak shift left behind (try 0.1) to test Re-zero")
    args = ap.parse_args()
    DISPLAY_HZ, SMOOTHING = args.hz, args.smoothing

    if args.check:
        if not args.si155:
            ap.error("--check needs --si155 <ip>")
        check_si155(args.si155)
        return
    if not args.sim and not args.si155:
        print("No --si155 address given, so starting the replay (same as --sim).\n"
              "For the real sensor:  python live_force.py --si155 10.0.0.55")
        args.sim = True
    models, inputs = load_models(args.models)
    if len(inputs) > len(FIBRE_MAP):
        sys.exit(f"The models expect {len(inputs)} inputs {inputs}, FIBRE_MAP has {len(FIBRE_MAP)}.")
    est = Estimator(models, inputs, args.model)
    print(f"Models: {', '.join(est.names)}; inputs {inputs}; showing {est.model_name}")
    src = (ReplaySource(args.dataset, inputs, seed=args.seed, hysteresis=args.hysteresis) if args.sim
           else SI155Source(args.si155, debug=args.debug, reconnect_every=args.reconnect_every))
    disp = LiveDisplay(src, est, log_path=args.log, show_band=not args.no_band, headless=args.frames > 0)
    if args.frames > 0:
        disp.run_headless(args.frames, args.save)
    else:
        disp.run()


if __name__ == "__main__":
    main()