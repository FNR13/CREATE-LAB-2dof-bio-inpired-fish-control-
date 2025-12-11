# Imports
import os
import sys

root_dir = os.path.abspath(os.curdir)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import math
import time
from kinematics import inverse_tail, dynamixel_angle_to_position, dynamixel_position_to_angle, fin_to_servo
from logger import DataLogger, plot_log

# -------------------------------
# Parameters
# -------------------------------
MODE = "symmetric_sin"   # "test", "symmetric_sin", or "standby"

# For MODE == "test"
PHI_TAIL = 15.0  # degrees
PHI_FIN = 10.0   # degrees

# For MODE == "symmetric_sin"
AMPLITUDE_TAIL = 20.0    # deg
AMPLITUDE_FIN = 15.0     # deg
FREQUENCY = 0.5          # Hz
PHASE = 0.0              # radians

LOOP_FREQUENCY = 100.0       # Hz
LOOP_DT = 1 / LOOP_FREQUENCY # seconds

LOG = True
LOG_FILENAME = "fish_robot_test_loop_log.xlsx"
PRINT = False
# -------------------------------

# --- Create logger ---
log_dir = os.path.join(root_dir, 'logs', LOG_FILENAME)
logger = DataLogger(log_dir)

if os.path.exists(log_dir):
    os.remove(log_dir)
    print(f"[INFO] Existing log file '{log_dir}' deleted.")

# --- Control loop simulation (no hardware) ---
print("[TEST] Starting control loop simulation (no hardware)...")

t0 = time.perf_counter()
next_time = t0

try:
    while True:
        # Wait until next loop interval
        now = time.perf_counter()
        sleep_time = next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_time += LOOP_DT

        # Compute elapsed time since start
        t = time.perf_counter() - t0

        # --- Compute desired fin/tail deflection ---
        if MODE == "test":
            phi_tail = PHI_TAIL
            phi_fin = PHI_FIN
        elif MODE == "symmetric_sin":
            phi_tail = AMPLITUDE_TAIL * math.sin(2 * math.pi * FREQUENCY * t)
            phi_fin  = AMPLITUDE_FIN * math.sin(2 * math.pi * FREQUENCY * t + PHASE)
        else:
            phi_tail = 0.0
            phi_fin = 0.0

        # --- Convert to servo angles ---
        theta_tail = inverse_tail(phi_tail)
        theta_fin = fin_to_servo(phi_fin)

        dynamixel_angle = 0

        # --- Save data / print ---
        if MODE == "test":
            break

        if PRINT:
            print(f"[t={t:6.2f}s] Tail: {phi_tail:+6.2f}° → {theta_tail:+6.2f}°,  "
                  f"Fin: {phi_fin:+6.2f}° → {theta_fin:+6.2f}°")

        if LOG:
            logger.log(
                t,
                phi_tail, theta_tail, dynamixel_angle,
                phi_fin, theta_fin
            )

except KeyboardInterrupt:
    print("\n[TEST] Simulation interrupted by user.")

    if LOG:
        logger.save()
        print("[STOP] Log saved.")
        plot_log(log_dir)
