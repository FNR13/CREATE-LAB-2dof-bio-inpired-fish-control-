
import sys
import math
import time

# ---------------------------------------------------------------------
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMUNNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Configuration constants (replace these to tune behavior)
# ---------------------------------------------------------------------

MODE = "symmetric_sin"   # options: "test", "symmetric_sin", "standby"

# For MODE == "test"
PHI_TAIL = 0  # degrees
PHI_FIN = 0   # degrees

# For MODE == "symmetric_sin"
AMPLITUDE_TAIL = 20.0    # deg
AMPLITUDE_FIN = 10.0     # deg
FREQUENCY = 0.5       # Hz
PHASE = -50*math.pi/180   # radians

LOOP_FREQUENCY = 60.0      # Hz
LOOP_DT = 1/LOOP_FREQUENCY     

# ---------------------------------------------------------------------

# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from fish_control_comms import Fish_Control_Comms
from kinematics import inverse_tail, dynamixel_angle_to_position, dynamixel_position_to_angle, fin_to_servo

# ---------------------------------------------------------------------
# Initialize communication interface
# ---------------------------------------------------------------------
fish_robot = Fish_Control_Comms(
    arduino_port="COM17", arduino_baudrate=115200, 
    dynamixel_port="COM18", dynamixel_baudrate=57600, dynamixel_ID=1, dynamixel_velocity=310
)

fish_robot.connect_devices()
print("[INFO] Fish control communication initialized")
# ---------------------------------------------------------------------

def main():

    # Initial positions
    fish_robot.move_dynamixel(dynamixel_angle_to_position(0))
    fish_robot.set_PWM_Angle(fin_to_servo(0))
    time.sleep(1.0)

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
            dynamixel_position = fish_robot.get_dynamixel_position()

            # Debug print (you can remove this in production)
            print(f"[t={t:6.2f}s] TAIL: phi:{phi_tail:+6.2f}° → theta: {theta_tail:+6.2f}°,  REAL_THETA  {dynamixel_angle:+6.2f}; "
                f"FIN: phi:{phi_fin:+6.2f}° → theta: {theta_fin:+6.2f}°")
            
    except KeyboardInterrupt:
        phi_tail = 0.0
        phi_fin = 0.0

        theta_tail = inverse_tail(phi_tail)
        theta_fin = fin_to_servo(phi_fin)

        fish_robot.move_dynamixel(dynamixel_angle_to_position(theta_tail))
        fish_robot.set_PWM_Angle(theta_fin)

        print("[STOP] All actuators set to 0°.")

if __name__ == "__main__":
    main()