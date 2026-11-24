// indent_rig.ino
// Motion controller for the FBG indentation slide.
// Arduino Uno + CNC Shield V3 + TMC2209 (standalone STEP/DIR) on the X axis.
//
// It only moves when told to, and refuses any move whose target is outside
// the stored soft limits. The PC (collect.py) owns the sequence.
//
// Serial: 115200 baud, 8N1, one command per line, newline terminated.
// Positions are in microsteps. Positive = downward, toward the sensor.
//
//   ?        -> RIG pos min max stepUs     status
//   M n      -> DONE pos | ERR LIMIT pos | ABORT pos   relative move, n microsteps (signed)
//   G n      -> same as M                   absolute move to n
//   Z        -> OK 0                        zero the position counter here
//   L a b    -> OK L a b                    set soft limits (microsteps)
//   V us     -> OK V us                     step period in us (60 to 10000)
//   E 0|1    -> OK E n                      driver disable / enable
//   P        -> POS pos                     print position
//   !        -> ABORT pos                   stop a move in progress (checked every microstep)
//
// Anything else -> ERR CMD

// ---------------- CONFIG ----------------
const bool INVERT_DIR = true;     // set true on the lab rig (Nov 2026): with false, "M 1600" moved UP
const uint8_t PIN_STEP = 2;       // CNC Shield V3, X axis
const uint8_t PIN_DIR  = 5;
const uint8_t PIN_EN   = 8;       // shared enable, active LOW

const long DEFAULT_LIM_MIN = 0;
const long DEFAULT_LIM_MAX = 24000;   // 15 rev at 1600 microsteps/rev
const unsigned int DEFAULT_STEP_US = 500;

const long RAMP_STEPS = 300;          // linear ramp over first and last 300 steps
const unsigned int RAMP_EXTRA_US = 1500;  // extra delay at the very start/end of a move
const unsigned int PULSE_US = 5;      // STEP high time (TMC2209 needs > 100 ns)
// ----------------------------------------

long pos = 0;                 // signed 32-bit, current position in microsteps
long limMin = DEFAULT_LIM_MIN;
long limMax = DEFAULT_LIM_MAX;
unsigned int stepUs = DEFAULT_STEP_US;
bool enabled = true;

char buf[40];
uint8_t bufLen = 0;

void setDriver(bool on) {
  enabled = on;
  digitalWrite(PIN_EN, on ? LOW : HIGH);
}

// Returns true if aborted by '!'. Any other character received mid-move is discarded.
bool abortRequested() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '!') return true;
  }
  return false;
}

// Move to an absolute target. Assumes the target has already been limit-checked.
void moveTo(long target) {
  long n = target - pos;
  if (n == 0) {
    Serial.print(F("DONE ")); Serial.println(pos);
    return;
  }
  bool forward = n > 0;
  long total = forward ? n : -n;
  bool dirLevel = forward ^ INVERT_DIR;
  digitalWrite(PIN_DIR, dirLevel ? HIGH : LOW);
  delayMicroseconds(20);  // DIR setup time

  if (!enabled) setDriver(true);

  for (long i = 0; i < total; i++) {
    if (abortRequested()) {
      Serial.print(F("ABORT ")); Serial.println(pos);
      return;
    }

    // Linear ramp: extra delay near the start and end of the move.
    long d = min(i, total - 1 - i);
    unsigned long extra = 0;
    if (d < RAMP_STEPS) {
      extra = (unsigned long)RAMP_EXTRA_US * (unsigned long)(RAMP_STEPS - d) / RAMP_STEPS;
    }
    unsigned long period = (unsigned long)stepUs + extra;

    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(PIN_STEP, LOW);
    pos += forward ? 1 : -1;

    unsigned long rest = period > PULSE_US ? period - PULSE_US : 0;
    // delayMicroseconds is only accurate up to ~16383 us; period is at most 11500 here.
    delayMicroseconds((unsigned int)rest);
  }
  Serial.print(F("DONE ")); Serial.println(pos);
}

void requestMove(long target) {
  if (target < limMin || target > limMax) {
    Serial.print(F("ERR LIMIT ")); Serial.println(pos);
    return;
  }
  moveTo(target);
}

void handleLine(char *line) {
  // skip leading spaces
  while (*line == ' ') line++;
  if (*line == '\0') return;

  char cmd = line[0];
  char *args = line + 1;
  char *end;

  switch (cmd) {
    case '?':
      Serial.print(F("RIG ")); Serial.print(pos); Serial.print(' ');
      Serial.print(limMin); Serial.print(' '); Serial.print(limMax); Serial.print(' ');
      Serial.println(stepUs);
      break;

    case 'M': case 'm': {
      long n = strtol(args, &end, 10);
      if (end == args) { Serial.println(F("ERR CMD")); break; }
      requestMove(pos + n);
      break;
    }

    case 'G': case 'g': {
      long n = strtol(args, &end, 10);
      if (end == args) { Serial.println(F("ERR CMD")); break; }
      requestMove(n);
      break;
    }

    case 'Z': case 'z':
      pos = 0;
      Serial.println(F("OK 0"));
      break;

    case 'L': case 'l': {
      long a = strtol(args, &end, 10);
      if (end == args) { Serial.println(F("ERR CMD")); break; }
      char *args2 = end;
      long b = strtol(args2, &end, 10);
      if (end == args2 || a > b) { Serial.println(F("ERR CMD")); break; }
      limMin = a; limMax = b;
      Serial.print(F("OK L ")); Serial.print(limMin); Serial.print(' '); Serial.println(limMax);
      break;
    }

    case 'V': case 'v': {
      long us = strtol(args, &end, 10);
      if (end == args || us < 60 || us > 10000) { Serial.println(F("ERR CMD")); break; }
      stepUs = (unsigned int)us;
      Serial.print(F("OK V ")); Serial.println(stepUs);
      break;
    }

    case 'E': case 'e': {
      long e = strtol(args, &end, 10);
      if (end == args || (e != 0 && e != 1)) { Serial.println(F("ERR CMD")); break; }
      setDriver(e == 1);
      Serial.print(F("OK E ")); Serial.println(enabled ? 1 : 0);
      break;
    }

    case 'P': case 'p':
      Serial.print(F("POS ")); Serial.println(pos);
      break;

    default:
      Serial.println(F("ERR CMD"));
  }
}

void setup() {
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR, OUTPUT);
  pinMode(PIN_EN, OUTPUT);
  digitalWrite(PIN_STEP, LOW);
  setDriver(true);
  Serial.begin(115200);
  Serial.println(F("READY"));
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '!') {                 // abort when idle: just report position
      Serial.print(F("ABORT ")); Serial.println(pos);
      bufLen = 0;
      continue;
    }
    if (c == '\r') continue;
    if (c == '\n') {
      buf[bufLen] = '\0';
      handleLine(buf);
      bufLen = 0;
      continue;
    }
    if (bufLen < sizeof(buf) - 1) buf[bufLen++] = c;
  }
}
