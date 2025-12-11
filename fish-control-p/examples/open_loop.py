import os
import sys

import math
import time


root_dir = os.path.abspath(os.curdir)
sys.path.insert(0, root_dir)

from support_scripts_py.fish_control_comms import Fish_Control_Comms 
from support_scripts_py.kinematics import inverse_tail, dynamixel_angle_to_position, dynamixel_position_to_angle, fin_to_servo
from support_scripts_py.logger import DataLogger, plot_log

# --- Warning / instructions ---
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMMUNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST

# --- Parameters ---
MODE = "symmetric_sin"   # options: "test", "symmetric_sin", "standby"

# For MODE == "test"
PHI_TAIL = 0    # degrees
PHI_FIN  = 0    # degrees

# For MODE == "symmetric_sin"
AMPLITUDE_TAIL = 45     # deg
AMPLITUDE_FIN  = 20     # deg
FREQUENCY      = 0.50   # Hz
PHASE          = 90 * math.pi / 180   # radians

LOOP_FREQUENCY = 200.0      # Hz
LOOP_DT        = 1 / LOOP_FREQUENCY

LOG          = False
LOG_FILENAME = "fish_robot_log.xlsx"
PRINT        = False

# --- Initialize communication interface ---
fish_robot = Fish_Control_Comms(
    arduino_port="COM22",
    arduino_baudrate=115200,
    dynamixel_port="COM18",
    dynamixel_baudrate=57600,
    dynamixel_ID=1,
    dynamixel_velocity=310
)
fish_robot.connect_devices()
print("[INFO] Fish control communication initialized")

if LOG:
    log_dir = os.path.join(root_dir, 'logs', LOG_FILENAME)
    logger = DataLogger(log_dir)
    if os.path.exists(log_dir):
        os.remove(log_dir)
        print(f"[INFO] Existing log file '{log_dir}' deleted.")

# --- Initial actuator positions ---
fish_robot.move_dynamixel(dynamixel_angle_to_position(0))
fish_robot.set_PWM_Angle(fin_to_servo(0))
time.sleep(0.5)

# --- Control loop ---
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

        # Elapsed time
        t = time.perf_counter() - t0

        # --- Compute desired fin/tail deflection ---
        if MODE == "test":
            phi_tail = PHI_TAIL
            phi_fin  = PHI_FIN
        elif MODE == "symmetric_sin":
            phi_tail = AMPLITUDE_TAIL * math.sin(2 * math.pi * FREQUENCY * t)
            phi_fin  = AMPLITUDE_FIN  * math.sin(2 * math.pi * FREQUENCY * t + PHASE)
        else:
            phi_tail = 0.0
            phi_fin  = 0.0

        # --- Convert to servo angles ---
        theta_tail = inverse_tail(phi_tail)
        theta_fin  = fin_to_servo(phi_fin)

        # --- Send commands to hardware ---
        fish_robot.move_dynamixel(dynamixel_angle_to_position(theta_tail))
        fish_robot.set_PWM_Angle(theta_fin)

        dynamixel_angle = dynamixel_position_to_angle(fish_robot.get_dynamixel_position())

        # --- Save / print data ---
        if MODE == "test":
            break

        if PRINT:
            print(f"[t={t:6.2f}s] TAIL: phi:{phi_tail:+6.2f}° → theta:{theta_tail:+6.2f}°, REAL_THETA:{dynamixel_angle:+6.2f}; "
                  f"FIN: phi:{phi_fin:+6.2f}° → theta:{theta_fin:+6.2f}°")

        if LOG:
            logger.log(
                t,
                phi_tail, theta_tail, dynamixel_angle,
                phi_fin, theta_fin
            )

except KeyboardInterrupt:
    # --- Stop actuators safely ---
    phi_tail = 0.0
    phi_fin  = 0.0
    theta_tail = inverse_tail(phi_tail)
    theta_fin  = fin_to_servo(phi_fin)

    fish_robot.move_dynamixel(dynamixel_angle_to_position(theta_tail))
    fish_robot.set_PWM_Angle(theta_fin)

    print("[STOP] All actuators set to 0°.")
    time.sleep(0.5)

    if LOG:
        logger.save()
        print("[STOP] Log saved.")
        plot_log(LOG_FILENAME)
