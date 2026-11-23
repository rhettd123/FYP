#!/usr/bin/env python3
"""
collect.py - runs the calibration rig and logs the data.

It controls three instruments from the PC:
  * the Arduino running indent_rig.ino, which moves the slide
  * the balance on COM12, which measures the load
  * the si155 interrogator, which reads the fibre wavelengths

Each run is one pose (x angle = shaft rotation, y angle = holder tilt). It is saved as
data/run_<time>_<block>_x<X>_y<Y>.csv, plus a _meta.json file with the settings used.
<block> is the first word of the note typed in at the start of the run.

A run goes like this:
  1. Tare the balance with the tip clear of the pan and log a zero-load row.
  2. Lower the tip until it just touches the pan. This point becomes 0 rev.
  3. Press down in small steps to 1.5 rev, or until the load limit.
  4. Come back up through the same steps and finish clear of the pan.
The slide has some backlash, so it only changes direction once, at the bottom.
A "thermal" run doesn't move; it just logs at a fixed position.

Usage:
  python collect.py --sim                     # no hardware, simulated instruments
  python collect.py                           # lab PC (Arduino COM11, balance COM12)
  python collect.py --arduino COM11 --balance COM12 --si155 10.0.0.55

All the settings are in the CONFIG block below.
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import socket
import struct
import sys
import threading
import time
from collections import deque

# =============================== CONFIG ===============================
# Motion
MICROSTEPS = 8                  # 1/8 microstepping, so 1600 steps per rev
FULL_STEPS_PER_REV = 200        # 1.8 deg motor
SLIDE_TOTAL_REV = 20            # length of the slide (for reference only)
# 0 rev is where the tip first touches the pan. It is found at the start of each run.
TRAVEL_MAX_REV = 1.5            # how far to press past first touch
LIFT_REV = 0.5                  # finish each run this far above the touch point
TRAVEL_MIN_REV = -LIFT_REV      # soft limit above the touch point
CONTACT_STEP_REV = 0.075        # step size while pressing
LEAD_MM_PER_REV = None          # screw lead in mm (None = not measured)
TOUCH_G = 1.0                   # load that counts as touching
TOUCH_SEARCH_STEP_REV = 0.015  # step size for the automatic touch search
TOUCH_SEARCH_MAX_REV = 3.0      # give up the touch search after this far
JOG_MAX_REV = 5.0               # largest manual jog allowed
BACKLASH_REV = 0.2              # free play when the slide changes direction
STEP_PERIOD_US = 500            # motor speed (raise it if the motor stalls)
MAX_LOAD_G = 40.0              # stop pressing at this load
MAX_STEPS_PER_RUN = 200         # safety cap on steps per run
LEVER_ARM_MM = 15.0             # tip to grating centre, saved in the meta file only
THERMAL_ROWS = 60               # rows logged in a thermal run
THERMAL_INTERVAL_S = 5.0        # seconds between thermal rows

# Balance. The lab balance sends its reading non-stop and ignores commands, so the
# script decides when the reading is stable and does the tare itself. "auto" checks
# which kind of balance is connected.
BALANCE_PROTOCOL = "auto"       # "auto", "stream" or "sics"
BALANCE_PORT = "COM12"
BALANCE_BAUD = 9600
BALANCE_BYTESIZE = 8
BALANCE_PARITY = "N"            # "O", "E" or "N"
BALANCE_STOPBITS = 1
BALANCE_XONXOFF = False         # only if the balance menu says Xon/Xoff
BALANCE_READABILITY_G = 0.1     # 0.1 g, saved in the meta file
SETTLE_S = 2.0                  # wait after each move before reading
STABLE_TIMEOUT_S = 8.0          # give up waiting for a stable reading after this
TARE_TIMEOUT_S = 15.0           # give up on the tare after this
TARE_AT_START = True            # tare at the start of each run
STABLE_WINDOW_S = 1.0           # stable = every reading over the last second is
STABLE_BAND_G = 0.2             #   within 0.2 g of the others,
STABLE_BAND_FRAC = 0.002        #   or within 0.2% of the reading if that's bigger
STREAM_STALE_S = 2.0            # no reading for this long = error

# Interrogator
SI155_IP = "10.0.0.55"
PEAK_SAMPLES = 50               # wavelength readings averaged per row
SI155_CHANNELS = (1, 2, 3, 4)
SI155_TIMEOUT_S = 2.0           # retry a read after this long
SI155_ATTEMPTS = 3              # tries per row before giving up
# fibre name -> (interrogator channel, peak number)
FIBRE_MAP = {
    "f1": (1, 0),
    "f2": (2, 0),
    "f3": (3, 0),
    # "temp": (4, 0),           # add once a fourth fibre is fitted
}

# Arduino
ARDUINO_PORT = "COM11"
ARDUINO_BAUD = 115200
ARDUINO_RESET_WAIT_S = 2.5      # the Uno restarts when the port opens

# Behaviour
LOG_RETURN_PATH = True          # also log the unloading path
SERIAL_ECHO = False             # print all serial traffic (for debugging)
DATA_DIR = "data"
# ======================================================================

STEPS_PER_REV = FULL_STEPS_PER_REV * MICROSTEPS
G = 9.80665


def rev_to_steps(rev):
    return int(round(rev * STEPS_PER_REV))


def steps_to_rev(steps):
    return steps / STEPS_PER_REV


def now_iso():
    return dt.datetime.now().isoformat(timespec="milliseconds")


def echo(prefix, text):
    if SERIAL_ECHO:
        print(f"    [{prefix}] {text!r}")


class RigError(Exception):
    pass


# ============================== ARDUINO ===============================
class Arduino:
    def __init__(self, port):
        import serial  # pyserial
        self.ser = serial.Serial(port, ARDUINO_BAUD, timeout=0.2)
        time.sleep(ARDUINO_RESET_WAIT_S)
        self.ser.reset_input_buffer()
        reply = self.command("?", timeout=3.0)
        if not reply.startswith("RIG"):
            raise RigError(f"Arduino did not answer '?' correctly: {reply!r}")

    def _write(self, line):
        echo("ARD >", line)
        self.ser.write((line + "\n").encode("ascii"))

    def _readline(self, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            raw = self.ser.readline()
            if raw:
                line = raw.decode("ascii", errors="replace").strip()
                echo("ARD <", line)
                if line and line != "READY":
                    return line
        raise TimeoutError("No reply from Arduino")

    def command(self, line, timeout=2.0):
        self._write(line)
        return self._readline(timeout)

    def status(self):
        parts = self.command("?").split()
        return {"pos": int(parts[1]), "min": int(parts[2]), "max": int(parts[3]),
                "step_us": int(parts[4])}

    def zero(self):
        return self.command("Z")

    def set_limits(self, lo, hi):
        return self.command(f"L {lo} {hi}")

    def set_speed(self, us):
        return self.command(f"V {us}")

    def enable(self, on):
        return self.command(f"E {1 if on else 0}")

    def _move_timeout(self, n_steps, step_us):
        return abs(n_steps) * step_us * 1e-6 + 600 * 1.5e-3 + 5.0

    def _parse_move(self, reply):
        parts = reply.split()
        if parts[0] == "DONE":
            return "DONE", int(parts[1])
        if parts[0] == "ERR" and parts[1] == "LIMIT":
            return "LIMIT", int(parts[2])
        if parts[0] == "ABORT":
            return "ABORT", int(parts[1])
        raise RigError(f"Unexpected move reply: {reply!r}")

    def move_to(self, target_steps, current_steps):
        timeout = self._move_timeout(target_steps - current_steps, STEP_PERIOD_US)
        self._write(f"G {target_steps}")
        try:
            return self._parse_move(self._readline(timeout))
        except KeyboardInterrupt:
            self.abort()
            raise

    def abort(self):
        echo("ARD >", "!")
        self.ser.write(b"!")
        try:
            reply = self._readline(2.0)
            return self._parse_move(reply)
        except Exception:
            return ("ABORT", None)

    def close(self):
        self.ser.close()


# ============================== BALANCE ===============================
# Commands for a Mettler balance that accepts them (MT-SICS):
#   SI = read the weight, T = tare, @ = reset and get the serial number, I2 = model
_SICS_WEIGHT_RE = re.compile(r"^S\s+([SD])\s+([+-]?\s*\d+(?:\.\d+)?)\s+(\S+)\s*$")
_UNIT_TO_G = {"g": 1.0, "mg": 1e-3, "kg": 1e3}


def parse_sics_weight(raw):
    """Read a weight reply. Returns (weight in g or None, stable, status)."""
    text = raw.strip()
    m = _SICS_WEIGHT_RE.match(text)
    if m:
        status, num, unit = m.groups()
        factor = _UNIT_TO_G.get(unit.lower())
        if factor is None:
            raise RigError(f"Balance is set to unit '{unit}'. Set it to grams in the balance menu.")
        return float(num.replace(" ", "")) * factor, status == "S", status
    parts = text.split()
    if len(parts) >= 2 and parts[0] == "S" and parts[1] in ("+", "-", "I"):
        return None, False, parts[1]
    return None, False, "ERR"


class Balance:
    """A Mettler balance that accepts commands."""

    def __init__(self, port):
        import serial
        parity = {"O": serial.PARITY_ODD, "E": serial.PARITY_EVEN,
                  "N": serial.PARITY_NONE}[BALANCE_PARITY.upper()]
        self.ser = serial.Serial(port, BALANCE_BAUD, bytesize=BALANCE_BYTESIZE,
                                 parity=parity, stopbits=BALANCE_STOPBITS,
                                 xonxoff=BALANCE_XONXOFF, timeout=1.0)
        time.sleep(0.3)
        self.ser.reset_input_buffer()

        # '@' resets the balance and returns its serial number.
        reply = self.command("@", timeout=3.0)
        if not reply.startswith("I4"):
            raise RigError(f"Balance did not answer MT-SICS '@' (got {reply!r}). "
                           "Check port, baud, data bits and parity against the balance menu.")
        self.serial_no = self._quoted(reply)
        self.model = self._quoted(self.command("I2")) or "unknown"

        w, _, status, raw = self.read_once()
        if w is None and status not in ("+", "-", "I"):
            raise RigError(f"Balance did not return a parseable weight: {raw!r}")

    @staticmethod
    def _quoted(reply):
        m = re.search(r'"(.*)"', reply)
        return m.group(1).strip() if m else ""

    def _readline(self, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            raw = self.ser.readline()
            if raw:
                line = raw.decode("ascii", errors="replace").strip()
                echo("BAL <", line)
                if line:
                    return line
        return ""

    def command(self, cmd, timeout=2.0):
        self.ser.reset_input_buffer()
        echo("BAL >", cmd)
        self.ser.write((cmd + "\r\n").encode("ascii"))
        return self._readline(timeout)

    def info(self):
        return {"model": self.model, "serial_no": self.serial_no, "protocol": "MT-SICS",
                "readability_g": BALANCE_READABILITY_G}

    def read_once(self):
        """One reading. Returns (weight_g, stable, status, raw)."""
        raw = self.command("SI")
        w, stable, status = parse_sics_weight(raw)
        return w, stable, status, raw

    def read_stable(self, timeout=STABLE_TIMEOUT_S, since=None):
        """Keep reading until the balance says it's stable, or until the timeout.
        Returns (weight_g, stable, raw)."""
        t_end = time.time() + timeout
        last = (None, False, "")
        while time.time() < t_end:
            w, stable, status, raw = self.read_once()
            if status == "+":
                raise RigError(f"Balance overload: {raw!r}")
            if w is not None:
                last = (w, stable, raw)
                if stable:
                    return last
            time.sleep(0.1)
        return last

    def tare(self):
        """Tare the balance, retrying while it's busy."""
        t_end = time.time() + TARE_TIMEOUT_S
        while time.time() < t_end:
            reply = self.command("T", timeout=max(1.0, t_end - time.time()))
            if reply.startswith("T S"):
                return True
            if reply.startswith(("T +", "T -")):
                raise RigError(f"Tare refused, load out of range: {reply!r}")
            time.sleep(0.5)             # balance busy or not settled yet
        print("  WARNING: tare did not complete (balance never settled).")
        return False

    def close(self):
        self.ser.close()


_STREAM_NUM_RE = re.compile(r"([+-]?)\s*(\d+(?:\.\d+)?)")


def parse_stream_line(text):
    """Turn a line like '   12.3' into 12.3 (None if there's no number)."""
    m = _STREAM_NUM_RE.search(text)
    if not m:
        return None
    v = float(m.group(2))
    return -v if m.group(1) == "-" else v


class StreamBalance:
    """A balance that sends its reading non-stop and takes no commands. A background
    thread keeps the recent readings. The reading counts as stable once it stops
    changing. The tare is done here by subtracting an offset, so the balance's own
    display won't read zero."""

    def __init__(self, port, ser=None):
        if ser is None:
            import serial
            parity = {"O": serial.PARITY_ODD, "E": serial.PARITY_EVEN,
                      "N": serial.PARITY_NONE}[BALANCE_PARITY.upper()]
            ser = serial.Serial(port, BALANCE_BAUD, bytesize=BALANCE_BYTESIZE, parity=parity,
                                stopbits=BALANCE_STOPBITS, xonxoff=BALANCE_XONXOFF, timeout=0.2)
        self.ser = ser
        self.port = port
        self.offset = 0.0
        self.buf = deque()                  # (time, reading in g, raw text)
        self.lock = threading.Lock()
        self.n_lines = self.n_bad = 0
        self._stop = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        t_end = time.time() + 3.0
        while time.time() < t_end and not self.buf:
            time.sleep(0.05)
        if not self.buf:
            self.close()
            raise RigError(f"No readings from the balance on {port} in 3 s. Check it is on and "
                           "that its output is set to continuous / stream.")

    def _reader(self):
        pending = b""
        while not self._stop:
            try:
                chunk = self.ser.read(256)
            except Exception:
                if self._stop:
                    return
                time.sleep(0.2)
                continue
            if not chunk:
                continue
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                text = line.decode("ascii", errors="replace").strip()
                if not text:
                    continue
                echo("BAL <", text)
                v = parse_stream_line(text)
                now = time.time()
                with self.lock:
                    self.n_lines += 1
                    if v is None:
                        self.n_bad += 1
                        continue
                    self.buf.append((now, v, text))
                    while self.buf and now - self.buf[0][0] > 30.0:
                        self.buf.popleft()

    def _window(self, since):
        with self.lock:
            return [(t, v, r) for t, v, r in self.buf if t >= since]

    def _latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def _check_fresh(self):
        last = self._latest()
        if last is None or time.time() - last[0] > STREAM_STALE_S:
            raise RigError(f"The balance stopped sending readings (nothing for {STREAM_STALE_S:g} s).")
        return last

    def _stable_now(self, since=None):
        now = time.time()
        start = now - STABLE_WINDOW_S
        if since is not None and since > start:
            return None
        win = self._window(start)
        if len(win) < 3:
            return None
        vals = [v for _, v, _ in win]
        band = max(STABLE_BAND_G, STABLE_BAND_FRAC * abs(vals[-1] - self.offset))
        if max(vals) - min(vals) <= band:
            return sum(vals) / len(vals), win[-1][2]
        return None

    def info(self):
        return {"model": "continuous-output balance", "serial_no": "", "protocol": "stream",
                "readability_g": BALANCE_READABILITY_G, "software_tare_g": round(self.offset, 3),
                "stable_rule": f"max({STABLE_BAND_G} g, {STABLE_BAND_FRAC:.1%} of reading) over {STABLE_WINDOW_S} s"}

    def read_once(self):
        """Latest reading. Returns (weight_g, stable, status, raw)."""
        t, v, raw = self._check_fresh()
        st = self._stable_now()
        return v - self.offset, st is not None, "S" if st is not None else "D", raw

    def read_stable(self, timeout=STABLE_TIMEOUT_S, since=None):
        """Wait until the reading is stable, using only readings since `since`.
        Returns (weight_g, stable, raw)."""
        t_call = time.time()
        t0 = t_call if since is None else min(since, t_call)
        while time.time() - t_call < max(timeout, STABLE_WINDOW_S + 0.5):
            self._check_fresh()
            st = self._stable_now(since=t0)
            if st is not None:
                return st[0] - self.offset, True, st[1]
            time.sleep(0.05)
        t, v, raw = self._check_fresh()
        return v - self.offset, False, raw

    def tare(self):
        """Wait for a stable reading and subtract it from then on."""
        old = self.offset
        self.offset = 0.0
        w, stable, _ = self.read_stable(TARE_TIMEOUT_S)
        if not stable:
            self.offset = old
            print("  WARNING: tare not done, the reading never settled. Keep the bench still.")
            return False
        self.offset = w
        return True

    def close(self):
        self._stop = True
        try:
            self.ser.close()
        except Exception:
            pass


def detect_balance_protocol(port, listen_s=1.5):
    """Listen for a moment. If numbers arrive on their own, it's a streaming balance."""
    import serial
    parity = {"O": serial.PARITY_ODD, "E": serial.PARITY_EVEN,
              "N": serial.PARITY_NONE}[BALANCE_PARITY.upper()]
    with serial.Serial(port, BALANCE_BAUD, bytesize=BALANCE_BYTESIZE, parity=parity,
                       stopbits=BALANCE_STOPBITS, xonxoff=BALANCE_XONXOFF, timeout=0.2) as ser:
        t_end = time.time() + listen_s
        got = b""
        while time.time() < t_end:
            got += ser.read(256)
    lines = [l for l in got.replace(b"\r", b"\n").split(b"\n") if l.strip()]
    nums = [l for l in lines if parse_stream_line(l.decode("ascii", "replace")) is not None]
    return "stream" if len(nums) >= 3 else "sics"


# ============================ INTERROGATOR ============================
class Si155Client:
    """Simple client for talking to the si155 over the network. Luna's own Python library
    kept opening new connections without closing the old ones, so the si155 stopped
    answering after about 10 reads. This one uses a single connection, gives up on a
    read that takes too long, and closes cleanly. live_force.py uses the same client."""

    COMMAND_PORT = 51971

    def __init__(self, ip, timeout=SI155_TIMEOUT_S):
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
        c, a = cmd.encode("ascii"), arg.encode("ascii")
        self.sock.sendall(struct.pack("<BBHI", 0, 0, len(c), len(a)) + c + a)
        status, _rtype, mlen, clen = struct.unpack("<BBHI", self._recv(8))
        msg = self._recv(mlen) if mlen else b""
        if status != 0:
            raise RigError(f"si155 refused {cmd}: {msg.decode('ascii', 'replace')}")
        return self._recv(clen) if clen else b""

    def peaks(self):
        """Peak wavelengths (nm) for each channel."""
        raw = self.command("#GetPeaks")
        length = struct.unpack("<HHIQII", raw[:24])[0]
        n_ch = (length - 24) // 2
        counts = struct.unpack(f"<{n_ch}h", raw[24:length])
        total = sum(counts)
        data = struct.unpack(f"<{total}d", raw[length:length + 8 * total])
        out, start = {}, 0
        for ch, n in enumerate(counts, start=1):
            out[ch] = list(data[start:start + n])
            start += n
        return out

    def serial_number(self):
        return self.command("#GetSerialNumber").decode("ascii", "replace").strip("\x00 ")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class Interrogator:
    """The si155, read through Si155Client. It connects fresh for each row, because the
    si155 drops connections left idle while the script waits for the user."""

    def __init__(self, ip):
        self.ip = ip
        self._cli = None
        self.connect()
        self.serial_no = self._cli.serial_number()
        self.read_peaks_once()
        self.disconnect()

    def connect(self):
        self.disconnect()
        try:
            self._cli = Si155Client(self.ip)
        except OSError as e:
            raise RigError(f"Cannot connect to the si155 at {self.ip}: {e}. Check the cable and "
                           "IP address, and close ENLIGHT or any other program using it.")

    def disconnect(self):
        if self._cli is not None:
            self._cli.close()
            self._cli = None

    def info(self):
        return {"ip": self.ip, "serial_no": self.serial_no, "client": "direct TCP (Si155Client)"}

    def read_peaks_once(self):
        peaks = self._cli.peaks()
        return {ch: [float(x) for x in peaks.get(ch, [])] for ch in SI155_CHANNELS}

    def read_averaged(self, n=PEAK_SAMPLES):
        for attempt in range(SI155_ATTEMPTS):
            try:
                self.connect()
                samples = [self.read_peaks_once() for _ in range(n)]
                return average_peaks(samples)
            except (OSError, ConnectionError, TimeoutError, struct.error) as e:
                print(f"  si155: {type(e).__name__}: {e} (attempt {attempt + 1} of {SI155_ATTEMPTS})")
                if attempt == SI155_ATTEMPTS - 1:
                    raise RigError(f"si155 not answering after {SI155_ATTEMPTS} attempts: {e}")
                time.sleep(0.5)
            finally:
                self.disconnect()

    def close(self):
        self.disconnect()


def average_peaks(samples):
    """Average a set of readings, skipping any with a missing or extra peak."""
    out = {}
    for ch in SI155_CHANNELS:
        lists = [s.get(ch, []) for s in samples]
        if not lists:
            out[ch] = []
            continue
        counts = [len(l) for l in lists]
        mode = max(set(counts), key=counts.count)
        good = [l for l in lists if len(l) == mode]
        out[ch] = [sum(l[i] for l in good) / len(good) for i in range(mode)]
    return out


# ============================= SIMULATION =============================
class SimWorld:
    """Fake rig for --sim. The tip starts 1 rev above the pan, and once it touches the
    load rises to about 300 g at 1.5 rev."""
    TOUCH_ABS_REV = 1.0
    BACKLASH_REV = 0.22         # free play, like the real slide

    def __init__(self):
        self._pos = 0               # motor position from the Arduino's zero
        self.zero_abs = 0           # where that zero is
        self.carriage = 0.0         # actual carriage position, rev
        self.tare_offset = 0.0
        self.x_deg = 0.0
        self.y_deg = 0.0

    @property
    def pos_steps(self):
        return self._pos

    @pos_steps.setter
    def pos_steps(self, v):
        self._pos = v
        m, b = steps_to_rev(self.zero_abs + v), self.BACKLASH_REV
        c = min(max(self.carriage, m - b), m)      # the carriage only moves once the play is taken up
        if self._load_at(m - b) > 0:               # while loaded, the sensor holds it against the screw
            c = m - b
        self.carriage = c

    def _load_at(self, c):
        d = c - self.TOUCH_ABS_REV
        return 0.0 if d <= 0 else 150.0 * d + 30.0 * d * d

    def load_g(self):
        return self._load_at(self.carriage)


class SimArduino:
    def __init__(self, world):
        self.w = world
        self.lo, self.hi = rev_to_steps(TRAVEL_MIN_REV), rev_to_steps(TRAVEL_MAX_REV)

    def status(self):
        return {"pos": self.w.pos_steps, "min": self.lo, "max": self.hi,
                "step_us": STEP_PERIOD_US}

    def zero(self):
        self.w.zero_abs += self.w._pos
        self.w._pos = 0
        return "OK 0"

    def set_limits(self, lo, hi):
        self.lo, self.hi = lo, hi
        return f"OK L {lo} {hi}"

    def set_speed(self, us):
        return f"OK V {us}"

    def enable(self, on):
        return f"OK E {1 if on else 0}"

    def move_to(self, target, current):
        if target < self.lo or target > self.hi:
            return "LIMIT", self.w.pos_steps
        time.sleep(0.05)
        self.w.pos_steps = target
        return "DONE", target

    def abort(self):
        return "ABORT", self.w.pos_steps

    def close(self):
        pass


class SimBalance:
    def __init__(self, world):
        self.w = world

    """Fake balance with 0.1 g readability."""

    def read_stable(self, timeout=STABLE_TIMEOUT_S, since=None):
        g = self.w.load_g() - self.w.tare_offset + random.gauss(0, 0.03)
        g = round(g / BALANCE_READABILITY_G) * BALANCE_READABILITY_G
        raw = f"S S {g:12.1f} g"
        return round(g, 1), True, raw

    def read_once(self):
        w, stable, raw = self.read_stable()
        return w, stable, "S", raw

    def info(self):
        return {"model": "SIM MT6001", "serial_no": "SIM", "protocol": "MT-SICS",
                "readability_g": BALANCE_READABILITY_G}

    def tare(self):
        self.w.tare_offset = self.w.load_g()
        return True

    def close(self):
        pass


class SimInterrogator:
    BASE = {1: [1550.0], 2: [1550.0], 3: [1550.0], 4: [1550.0]}   # all gratings near 1550 nm
    PHI = {1: 0.0, 2: 120.0, 3: 240.0}

    def __init__(self, world):
        self.w = world

    def info(self):
        return {"ip": "SIM", "serial_no": "SIM", "client": "simulated"}

    def read_averaged(self, n=PEAK_SAMPLES):
        F = self.w.load_g() * G / 1000.0
        th, ph = math.radians(self.w.y_deg), math.radians(self.w.x_deg)
        fz = F * math.cos(th)
        fx, fy = F * math.sin(th) * math.sin(ph), F * math.sin(th) * math.cos(ph)   # same convention as build_dataset.py
        out = {}
        for ch, base in self.BASE.items():
            if ch in self.PHI:
                p = math.radians(self.PHI[ch])
                shift_pm = 15.0 * fz + 60.0 * (fx * math.cos(p) + fy * math.sin(p))
            else:
                shift_pm = 0.5 * fz
            shift_pm += random.gauss(0, 0.3)
            out[ch] = [base[0] + shift_pm / 1000.0]
        return out

    def close(self):
        pass


# ============================== RUN LOGIC =============================
def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    ans = input(f"{prompt}{suffix}: ").strip()
    return ans if ans else (default if default is not None else "")


def ask_float(prompt):
    while True:
        try:
            return float(ask(prompt))
        except ValueError:
            print("  Please enter a number.")


def ask_yes(prompt, default="y"):
    return ask(prompt + " (y/n)", default).lower().startswith("y")


def angle_tag(v):
    return f"{v:g}".replace(".", "p").replace("-", "m")


def fibre_values(peaks):
    vals = {}
    for name, (ch, idx) in FIBRE_MAP.items():
        lst = peaks.get(ch, [])
        vals[name] = lst[idx] if idx < len(lst) else None
    return vals


def config_snapshot():
    keys = ["MICROSTEPS", "FULL_STEPS_PER_REV", "STEPS_PER_REV", "SLIDE_TOTAL_REV",
            "TRAVEL_MIN_REV", "TRAVEL_MAX_REV", "LIFT_REV", "LEAD_MM_PER_REV",
            "CONTACT_STEP_REV", "TOUCH_G", "TOUCH_SEARCH_STEP_REV", "TOUCH_SEARCH_MAX_REV",
            "STEP_PERIOD_US", "MAX_LOAD_G", "MAX_STEPS_PER_RUN", "LEVER_ARM_MM", "BACKLASH_REV",
            "THERMAL_ROWS", "THERMAL_INTERVAL_S", "BALANCE_PORT", "BALANCE_BAUD", "BALANCE_BYTESIZE",
            "BALANCE_PARITY", "BALANCE_STOPBITS", "BALANCE_XONXOFF",
            "BALANCE_READABILITY_G", "SETTLE_S", "STABLE_TIMEOUT_S", "TARE_TIMEOUT_S",
            "TARE_AT_START", "BALANCE_PROTOCOL", "STABLE_WINDOW_S", "STABLE_BAND_G", "STABLE_BAND_FRAC",
            "SI155_IP", "PEAK_SAMPLES", "SI155_CHANNELS", "SI155_TIMEOUT_S", "SI155_ATTEMPTS",
            "FIBRE_MAP", "LOG_RETURN_PATH"]
    g = globals()
    return {k: g[k] for k in keys}


class Run:
    def __init__(self, rig, x_deg, y_deg, note, thermal=False):
        self.rig = rig
        self.thermal = thermal
        self.x, self.y, self.note = x_deg, y_deg, note
        os.makedirs(DATA_DIR, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        # The block (first word of the note) goes in the file name, or "nonote" if blank.
        word = re.sub(r"[^A-Za-z0-9]", "", note.split()[0]) if note.strip() else ""
        base = f"run_{stamp}_{(word or 'nonote').lower()[:20]}_x{angle_tag(x_deg)}_y{angle_tag(y_deg)}"
        self.csv_path = os.path.join(DATA_DIR, base + ".csv")
        self.meta_path = os.path.join(DATA_DIR, base + "_meta.json")
        self.fields = (["timestamp", "x_angle_deg", "y_angle_deg", "note", "phase", "step_no",
                        "position_rev", "position_mm", "weight_g", "force_N",
                        "weight_stable"]
                       + [f"{n}_nm" for n in FIBRE_MAP]
                       + ["peak_samples", "all_peaks_json", "balance_raw"])
        self.fh = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=self.fields)
        self.writer.writeheader()
        self.rows = 0
        self.pos_steps = 0
        self.stop_reason = None
        self.meta = {"started": now_iso(), "x_angle_deg": x_deg, "y_angle_deg": y_deg,
                     "note": note, "thermal": thermal, "sim": rig["sim"], "ports": rig["ports"],
                     "balance": rig["balance"].info(),
                     "si155": rig["si155"].info(),
                     "config": config_snapshot()}

    def measure(self, phase, step_no, settle=True, defer=False):
        """Log one row. With defer=True the row waits until the touch point is known."""
        if settle:
            time.sleep(SETTLE_S if not self.rig["sim"] else min(SETTLE_S, 0.05))
        w, stable, raw = self.rig["balance"].read_stable(since=getattr(self, "last_move_t", None))
        peaks = self.rig["si155"].read_averaged()
        fib = fibre_values(peaks)
        rev = steps_to_rev(self.pos_steps)
        row = {
            "timestamp": now_iso(),
            "x_angle_deg": self.x, "y_angle_deg": self.y, "note": self.note,
            "phase": phase, "step_no": step_no,
            "position_rev": f"{rev:.4f}",
            "position_mm": f"{rev * LEAD_MM_PER_REV:.4f}" if LEAD_MM_PER_REV else "",
            "weight_g": "" if w is None else f"{w:.3f}",
            "force_N": "" if w is None else f"{w * G / 1000.0:.5f}",
            "weight_stable": 1 if stable else 0,
            "peak_samples": PEAK_SAMPLES,
            "all_peaks_json": json.dumps({str(k): [round(v, 5) for v in vals]
                                          for k, vals in peaks.items()}),
            "balance_raw": raw,
        }
        for n, v in fib.items():
            row[f"{n}_nm"] = "" if v is None else f"{v:.5f}"
        if defer:
            self.deferred = row
        else:
            self.writer.writerow(row)
            self.fh.flush()
            self.rows += 1
        fib_txt = " ".join(f"{n}_nm={'' if v is None else f'{v:.4f}'}" for n, v in fib.items())
        w_txt = "  ---  g" if w is None else f"{w:7.2f} g"
        flag = "" if stable else " (unstable)"
        pos_txt = "  clear" if defer else f"{rev:7.3f}"
        print(f"  {phase:<6} #{step_no:<3} pos {pos_txt} rev  {w_txt}{flag}  {fib_txt}")
        return w if w is not None else 0.0

    def write_deferred(self, rel_steps=None):
        """Write the held-back zero-load row, with its position from the touch point."""
        row = getattr(self, "deferred", None)
        if row is None:
            return
        if rel_steps is None:
            row["position_rev"] = row["position_mm"] = ""
        else:
            rev = steps_to_rev(rel_steps)
            row["position_rev"] = f"{rev:.4f}"
            row["position_mm"] = f"{rev * LEAD_MM_PER_REV:.4f}" if LEAD_MM_PER_REV else ""
        self.writer.writerow(row)
        self.fh.flush()
        self.rows += 1
        self.deferred = None

    def move(self, target_steps):
        ard = self.rig["arduino"]
        status, pos = ard.move_to(target_steps, self.pos_steps)
        if pos is not None:
            self.pos_steps = pos
        if self.rig["sim"]:
            self.rig["world"].pos_steps = self.pos_steps
        self.last_move_t = time.time() + 0.5     # the balance lags the move by about 0.5 s
        if status != "DONE":
            raise RigError(f"Move to {target_steps} returned {status} at {pos}")

    def execute(self):
        """Press from the touch point (0) down to TRAVEL_MAX_REV, then come back up.
        The zero-load row was logged before the touch point was found."""
        lift = -rev_to_steps(LIFT_REV)
        max_steps = rev_to_steps(TRAVEL_MAX_REV)
        step = rev_to_steps(CONTACT_STEP_REV)
        step_no = 0

        # Loading: always moving down, so backlash doesn't matter.
        visited = []
        target = 0
        while True:
            self.move(target)
            step_no += 1
            w = self.measure("load", step_no)
            visited.append(self.pos_steps)
            if w >= MAX_LOAD_G:
                self.stop_reason = "max_load"
                break
            if step_no >= MAX_STEPS_PER_RUN:
                self.stop_reason = "max_steps"
                break
            if self.pos_steps >= max_steps:
                self.stop_reason = "full_depth"
                break
            target = min(self.pos_steps + step, max_steps)

        print(f"  Loading stopped: {self.stop_reason} at {steps_to_rev(self.pos_steps):.3f} rev")

        # Unloading: the only change of direction. The loaded tip keeps the play taken up.
        if LOG_RETURN_PATH:
            for s in reversed(visited[:-1]):
                self.move(s)
                step_no += 1
                self.measure("unload", step_no)
            self.move(lift)
            step_no += 1
            w = self.measure("unload", step_no)
            if w > TOUCH_G:
                print(f"  WARNING: {w:.1f} g at the end, the tip did not clear the pan. "
                      f"Raise LIFT_REV (now {LIFT_REV:g}); the backlash may be more than {BACKLASH_REV:g} rev.")
        else:
            self.move(lift)

    def execute_thermal(self):
        """Log at a fixed position while the sensor warms and cools. Ctrl-C ends it early."""
        self.stop_reason = "thermal_complete"
        print(f"  Thermal run: {THERMAL_ROWS} rows every {THERMAL_INTERVAL_S:g} s. "
              f"Warm the sensor now; Ctrl-C to finish early.")
        try:
            for n in range(1, THERMAL_ROWS):
                t0 = time.time()
                self.measure("load", n, settle=False)
                wait = THERMAL_INTERVAL_S - (time.time() - t0)
                if wait > 0 and n < THERMAL_ROWS - 1:
                    time.sleep(wait if not self.rig["sim"] else 0.01)
        except KeyboardInterrupt:
            self.stop_reason = "thermal_stopped_early"
            print("\n  Thermal run ended early (rows kept).")

    def close(self, aborted=False):
        self.fh.close()
        self.meta.update({"finished": now_iso(), "rows": self.rows,
                          "stop_reason": "aborted" if aborted else self.stop_reason,
                          "final_position_steps": self.pos_steps})
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)
        print(f"  Wrote {self.rows} rows to {self.csv_path}")


def show_weight(rig):
    w, stable, status, _ = rig["balance"].read_once()
    return "  ---  g" if w is None else f"{w:7.2f} g{'' if stable else ' (moving)'}"


def jog(rig, pos, prompt, allow_find=False):
    """Move the slide by hand, showing the balance after each move. 'f' finds the touch
    point. Returns the new position in steps."""
    ard = rig["arduino"]
    while True:
        hint = ", f = find touch" if allow_find else ""
        j = ask(f"  {prompt} (motor turns, e.g. -0.5 = half a turn up, + = down{hint}, "
                f"blank = done)", "")
        if j == "":
            return pos
        if allow_find and j.lower() == "f":
            pos = find_touch(rig, pos)
            continue
        try:
            turns = float(j)
        except ValueError:
            print("  Please enter a number of motor turns, e.g. -0.5 or 0.1.")
            continue
        if abs(turns) > JOG_MAX_REV:
            print(f"  {turns:g} is motor turns (revolutions), not steps or mm. Jogs are limited to "
                  f"{JOG_MAX_REV:g} turns at a time; the whole slide is {SLIDE_TOTAL_REV:g}.")
            continue
        status, pos = ard.move_to(pos + rev_to_steps(turns), pos)
        if rig["sim"]:
            rig["world"].pos_steps = pos
        time.sleep(0.3 if not rig["sim"] else 0)
        if status == "LIMIT":
            print(f"  Not moved: more than {SLIDE_TOTAL_REV:g} turns from where this run started.")
        print(f"  {status} at {steps_to_rev(pos):+.3f} turns   balance {show_weight(rig)}")


def find_touch(rig, pos):
    """Step down until the balance reads TOUCH_G, so the tip is just touching."""
    ard = rig["arduino"]
    step = rev_to_steps(TOUCH_SEARCH_STEP_REV)
    limit = pos + rev_to_steps(TOUCH_SEARCH_MAX_REV)
    print(f"  Searching for touch in {TOUCH_SEARCH_STEP_REV} rev steps (Ctrl-C to stop) ...")
    try:
        while pos + step <= limit:
            status, pos = ard.move_to(pos + step, pos)
            if rig["sim"]:
                rig["world"].pos_steps = pos
            if status != "DONE":
                print(f"  Move stopped: {status}")
                return pos
            time.sleep(0.4 if not rig["sim"] else 0)
            w, _, _, _ = rig["balance"].read_once()
            if w is not None and w >= TOUCH_G:
                print(f"  Touching at {steps_to_rev(pos):+.3f} rev, balance {w:.2f} g")
                return pos
        print(f"  No touch within {TOUCH_SEARCH_MAX_REV} rev. Jog closer and try again.")
    except KeyboardInterrupt:
        _, p = ard.abort()
        if p is not None:
            pos = p
        print(f"\n  Search stopped at {steps_to_rev(pos):+.3f} rev.")
    return pos


def one_run(rig):
    print()
    x = ask_float("Shaft rotation, x angle (deg)")
    y = ask_float("Holder tilt, y angle (deg)")
    note = ask("Note (start with the block: grid, axial, repeat, ref, thermal ...)", "")
    thermal = note.strip().lower().startswith("thermal")
    if rig["sim"]:
        rig["world"].x_deg, rig["world"].y_deg = x, y

    ard = rig["arduino"]
    span = rev_to_steps(SLIDE_TOTAL_REV)
    ard.zero()
    ard.set_limits(-span, span)                 # loose limits during setup
    if rig["sim"]:
        rig["world"].pos_steps = 0
    pos = 0

    # 1. Tip clear of the pan: tare, then log the zero-load row.
    print(f"  Balance now {show_weight(rig)}. The tip must be clear of the pan.")
    while True:
        pos = jog(rig, pos, "Lift the tip clear if needed")
        if not rig.get("tared"):
            break                   # first run: the balance hasn't been tared yet
        w, _, _, _ = rig["balance"].read_once()
        if w is None or abs(w) <= TOUCH_G:
            break
        if ask_yes(f"  Balance reads {w:.1f} g. Is the tip definitely clear of the pan?", "n"):
            break
    if TARE_AT_START:
        rig["tared"] = rig["balance"].tare() or rig.get("tared", False)
        w, stable, _ = rig["balance"].read_stable()
        print(f"  After tare: {w} g{'' if stable else ' (unstable)'}")

    run = Run(rig, x, y, note, thermal=thermal)
    aborted = False
    lift = -rev_to_steps(LIFT_REV)
    zeroed = False
    try:
        run.measure("load", 0, defer=not thermal)
        baseline_pos = pos

        if thermal:
            run.execute_thermal()
            return

        # 2. Find the touch point (moving down only). This becomes 0 rev.
        while True:
            print(f"  Bring the tip down until it just touches the pan (about {TOUCH_G:g} g). "
                  f"Only jog down from here; if you overshoot, back off well past it and come down again.")
            pos = jog(rig, pos, "Move to touch", allow_find=True)
            w, _, _, _ = rig["balance"].read_once()
            print(f"  Balance {show_weight(rig)}")
            if w is not None and w > 10 * TOUCH_G:
                print("  That is well past just touching. Back off and try again.")
                continue
            if ask_yes("Is the tip just touching? This becomes 0 rev"):
                break
        ard.zero()
        if rig["sim"]:
            rig["world"].zero_abs += rig["world"]._pos
            rig["world"]._pos = 0
        ard.set_limits(rev_to_steps(TRAVEL_MIN_REV), rev_to_steps(TRAVEL_MAX_REV))
        run.pos_steps = 0
        zeroed = True
        run.write_deferred(baseline_pos - pos)
        print(f"  Zeroed at touch. Pressing to {TRAVEL_MAX_REV:g} rev in {CONTACT_STEP_REV:g} rev "
              f"steps, then back up to {-LIFT_REV:+g} rev.")

        # 3 and 4. Load and unload.
        run.execute()
    except KeyboardInterrupt:
        aborted = True
        print("\n  Ctrl-C: motor aborted.")
        status, p = rig["arduino"].abort()
        if p is not None:
            run.pos_steps = p
        try:
            if zeroed and run.pos_steps > lift and ask_yes("  Lift the tip clear?"):
                run.move(lift)
        except (KeyboardInterrupt, RigError):
            rig["arduino"].abort()
    except RigError as e:
        aborted = True
        print(f"  Stopped: {e}")
        try:
            if zeroed and run.pos_steps > lift and ask_yes("  Lift the tip clear?"):
                run.move(lift)
        except (KeyboardInterrupt, RigError) as e2:
            rig["arduino"].abort()
            print(f"  Could not lift clear: {e2}")
    finally:
        run.write_deferred(None)            # only if the run stopped before the touch point was found
        run.close(aborted)
        where = "where the thermal run started" if thermal else "this run's touch point"
        print(f"  Slide at {steps_to_rev(run.pos_steps):+.3f} rev from {where}.")


def connect(args):
    rig = {"sim": args.sim, "ports": {"arduino": args.arduino, "balance": args.balance,
                                      "si155": args.si155}}
    if args.sim:
        world = SimWorld()
        rig.update(world=world, arduino=SimArduino(world), balance=SimBalance(world),
                   si155=SimInterrogator(world))
        print("SIMULATION MODE: no hardware.")
        return rig

    print(f"Connecting to Arduino on {args.arduino} ...")
    rig["arduino"] = Arduino(args.arduino)
    rig["arduino"].set_speed(STEP_PERIOD_US)
    rig["arduino"].enable(True)
    print(f"  OK {rig['arduino'].status()}")

    print(f"Connecting to balance on {args.balance} ...")
    proto = BALANCE_PROTOCOL
    if proto == "auto":
        proto = detect_balance_protocol(args.balance)
    if proto == "stream":
        rig["balance"] = StreamBalance(args.balance)
        w, _, _, raw = rig["balance"].read_once()
        print(f"  OK  continuous output ({raw!r}); stability judged from the stream, "
              f"tare done in software")
    else:
        rig["balance"] = Balance(args.balance)
        info = rig["balance"].info()
        print(f"  OK  MT-SICS, model '{info['model']}', serial {info['serial_no']}")

    print(f"Connecting to si155 at {args.si155} ...")
    rig["si155"] = Interrogator(args.si155)
    print(f"  OK  serial {rig['si155'].serial_no}")
    return rig


def main():
    global DATA_DIR
    p = argparse.ArgumentParser(description="FBG indentation rig data collection")
    p.add_argument("--arduino", default=ARDUINO_PORT)
    p.add_argument("--balance", default=BALANCE_PORT)
    p.add_argument("--si155", default=SI155_IP)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--sim", action="store_true", help="simulated instruments")
    args = p.parse_args()
    DATA_DIR = args.data_dir

    try:
        rig = connect(args)
    except Exception as e:
        print(f"Connection failed: {type(e).__name__}: {e}")
        sys.exit(1)

    try:
        while True:
            one_run(rig)
            if not ask_yes("Another pose?"):
                break
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    finally:
        for k in ("arduino", "balance", "si155"):
            try:
                rig[k].close()
            except Exception:
                pass


if __name__ == "__main__":
    main()