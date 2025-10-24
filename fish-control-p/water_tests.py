
import sys
import math
import time

# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from kinematics import fin_to_servo, inverse_tail 
from fish_control_comms import Fish_Control_Comms

# MAKE SURE TO CLOSE DYNAMIXEL WIZARD AND ARDUINO IDE BEFORE RUNNING THIS TEST
# END COMUNNICATIONS TO AVOID ISSUES IN RERUNNING THE TEST

def main():
    # ---------------------------------------------------------------------
    # Configuration constants (replace these to tune behavior)
    # ---------------------------------------------------------------------
    MODE = "symmetric_sin"   # options: "test", "symmetric_sin", "standby"

    # For MODE == "test"
    PHI_TAIL = 15.0  # degrees
    PHI_FIN = 10.0   # degrees

    # For MODE == "symmetric_sin"
    AMPLITUDE_TAIL = 20.0    # deg
    AMPLITUDE_FIN = 15.0     # deg
    FREQUENCY = 0.5          # Hz
    PHASE = 0.0              # radians

    LOOP_FREQUENCY = 20.0      # Hz
    LOOP_DT = 1/LOOP_FREQUENCY     

    # ---------------------------------------------------------------------
    # Initialize communication interface
    # ---------------------------------------------------------------------
    fish_robot = Fish_Control_Comms(
        arduino_port="COM17", arduino_baudrate=115200, 
        dynamixel_port="COM18", dynamixel_baudrate=57600, dynamixel_ID=1, dynamixel_velocity=80
    )

    print("[INFO] Fish control communication initialized")

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

        # -----------------------------------------------------------------
        # Send commands to hardware via comms interface
        # -----------------------------------------------------------------
        fish_robot.move_dynamixel(theta_tail)
        fish_robot.set_PWM_Angle(theta_fin)

        # Debug print (you can remove this in production)
        print(f"[t={t:6.2f}s] Tail: {phi_tail:+6.2f}° → {theta_tail:+6.2f}°,  "
              f"Fin: {phi_fin:+6.2f}° → {theta_fin:+6.2f}°")


if __name__ == "__main__":
    main()