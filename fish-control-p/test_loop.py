import sys
import math
import time

# Add your support folder to the path
sys.path.insert(0, 'support_scripts_py')

from kinematics import fin_to_servo, inverse_tail


def test_loop():
    """
    Test the timing and waveform calculations for tail and fin motions
    without sending commands to any hardware.
    """

    # ---------------------------------------------------------------------
    # Configuration constants
    # ---------------------------------------------------------------------
    MODE = "symmetric_sin"   # "test", "symmetric_sin", or "standby"

    # For MODE == "test"
    PHI_TAIL = 15.0  # degrees
    PHI_FIN = 10.0   # degrees

    # For MODE == "symmetric_sin"
    AMPLITUDE_TAIL = 20.0    # deg
    AMPLITUDE_FIN = 15.0     # deg
    FREQUENCY = 0.5          # Hz
    PHASE = 0.0              # radians

    LOOP_DT = 0.05           # seconds (20 Hz update rate)
    TEST_DURATION = 10.0     # seconds — stop automatically after this

    print("[TEST] Starting control loop simulation (no hardware)...")

    t0 = time.perf_counter()
    next_time = t0

    while True:
        # Wait until next loop interval
        now = time.perf_counter()
        sleep_time = next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_time += LOOP_DT

        # Compute elapsed time since start
        t = time.perf_counter() - t0
        if t > TEST_DURATION:
            print("[TEST] Simulation complete.")
            break

        # -----------------------------------------------------------------
        # Compute desired fin/tail deflection
        # -----------------------------------------------------------------
        if MODE == "test":
            phi_tail = PHI_TAIL
            phi_fin = PHI_FIN
        elif MODE == "symmetric_sin":
            phi_tail = AMPLITUDE_TAIL * math.sin(2 * math.pi * FREQUENCY * t)
            phi_fin  = AMPLITUDE_FIN * math.sin(2 * math.pi * FREQUENCY * t + PHASE)
        else:
            phi_tail = 0.0
            phi_fin = 0.0

        # -----------------------------------------------------------------
        # Convert to servo angles
        # -----------------------------------------------------------------
        theta_tail = inverse_tail(phi_tail)
        theta_fin = fin_to_servo(phi_fin)

        # Optional calibration override (for MODE == "test")
        if MODE == "test":
            theta_tail = phi_tail
            theta_fin = phi_fin

        # -----------------------------------------------------------------
        # Debug print (simulation output)
        # -----------------------------------------------------------------
        print(f"[t={t:6.2f}s] Tail: {phi_tail:+6.2f}° → {theta_tail:+6.2f}°,  "
              f"Fin: {phi_fin:+6.2f}° → {theta_fin:+6.2f}°")

    print("[TEST] Loop exited normally.")


if __name__ == "__main__":
    test_loop()
