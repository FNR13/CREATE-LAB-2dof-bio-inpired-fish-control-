import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import time
from fish_control_comms import Fish_Control_Comms
from kinematics import inverse_tail, dynamixel_angle_to_position

# --- Warning / instructions ---
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMMUNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST


# --- Initialize Fish Control ---
fish_robot = Fish_Control_Comms(
    arduino_port="COM17",
    arduino_baudrate=115200,
    dynamixel_port="COM18",
    dynamixel_baudrate=57600,
    dynamixel_ID=1,
    dynamixel_velocity=80
)

fish_robot.connect_devices()

# --- PWM Test Loop ---
print("Starting Fish Control Comms Test")

for i in range(5):
    fish_robot.set_PWM_Angle(90 + 2*i)
    print(f"Set PWM Angle to {90 + i}")
    time.sleep(0.5)

# --- Optional Dynamixel movement tests ---

# print("\nMoving Dynamixel to 0 degrees")
# value = dynamixel_angle_to_position(0)
# print("Dynamixel value:", value)
# fish_robot.move_dynamixel(value)
# input("Press Enter to continue...")

# print("\nMoving Dynamixel to 78 degrees")
# value = dynamixel_angle_to_position(78)
# print("Dynamixel value:", value)
# fish_robot.move_dynamixel(value)
# input("Press Enter to continue...")

# print("\nMoving Dynamixel to -78 degrees")
# value = dynamixel_angle_to_position(-78)
# print("Dynamixel value:", value)
# fish_robot.move_dynamixel(value)
