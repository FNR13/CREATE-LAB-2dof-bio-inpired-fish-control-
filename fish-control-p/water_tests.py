import os
import sys
import math
import time

# ---------------------------------------------------------------------
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMUNNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------

MODE = "symmetric_sin"   # options: "test", "symmetric_sin", "standby"

# For MODE == "test"
PHI_TAIL = 0  # degrees
PHI_FIN = 0 # degrees

# For MODE == "symmetric_sin"
AMPLITUDE_TAIL = 45    # deg
AMPLITUDE_FIN = 20  # deg
FREQUENCY = 0.50   # Hz
PHASE = 90*math.pi/180   # radians

LOOP_FREQUENCY =200.0      # Hz
LOOP_DT = 1/LOOP_FREQUENCY     

LOG = False
LOG_FILENAME = "fish_robot_log.xlsx"

PRINT = False

# ---------------------------------------------------------------------

# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from fish_control_comms import Fish_Control_Comms
from kinematics import inverse_tail, dynamixel_angle_to_position, dynamixel_position_to_angle, fin_to_servo
from logger import DataLogger, plot_log

# ---------------------------------------------------------------------
# Initialize communication interface
# ---------------------------------------------------------------------
fish_robot = Fish_Control_Comms(
    arduino_port="COM22", arduino_baudrate=115200, 
    dynamixel_port="COM18", dynamixel_baudrate=57600, dynamixel_ID=1, dynamixel_velocity=310
)

fish_robot.connect_devices()
print("[INFO] Fish control communication initialized")

if LOG:
    # Create the logger
    logger = DataLogger(LOG_FILENAME)

    # Delete previous file if it exists
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)
        print(f"[INFO] Existing log file '{LOG_FILENAME}' deleted.")

# ---------------------------------------------------------------------

def main():

    # Initial positions
    fish_robot.move_dynamixel(dynamixel_angle_to_position(0))
    fish_robot.set_PWM_Angle(fin_to_servo(0))
    time.sleep(3.0)

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
            fish_robot.move_dynamixel(dynamixel_angle_to_position(theta_tail))
            fish_robot.set_PWM_Angle(theta_fin)

            dynamixel_angle = dynamixel_position_to_angle(fish_robot.get_dynamixel_position())

            # -----------------------------------------------------------------
            # Save data
            if MODE=="test":
                break

            if PRINT:
                print(f"[t={t:6.2f}s] TAIL: phi:{phi_tail:+6.2f}° → theta: {theta_tail:+6.2f}°,  REAL_THETA  {dynamixel_angle:+6.2f}; "
                    f"FIN: phi:{phi_fin:+6.2f}° → theta: {theta_fin:+6.2f}°")

            if LOG:
                logger.log(
                            t,
                            phi_tail, theta_tail, dynamixel_angle,
                            phi_fin, theta_fin
                        )
                
    except KeyboardInterrupt:
        phi_tail = 0.0
        phi_fin = 0.0

        theta_tail = inverse_tail(phi_tail)
        theta_fin = fin_to_servo(phi_fin)

        fish_robot.move_dynamixel(dynamixel_angle_to_position(theta_tail))
        fish_robot.set_PWM_Angle(theta_fin)

        print("[STOP] All actuators set to 0°.")
        time.sleep(1.0)

        if LOG:
            logger.save()
            print("[STOP] Log saved.")

            plot_log(LOG_FILENAME)


if __name__ == "__main__":
    main()