# Imports
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

from dynamixel_controller import Dynamixel

# --- Warning / instructions ---
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMMUNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST


# --- Initialize Dynamixel ---
test_dynamixel = Dynamixel(
    ID=1,
    descriptive_device_name="Flag dynamixel",
    port_name="COM19",
    baudrate=57600
)

test_dynamixel.begin_communication()
test_dynamixel.set_operating_mode("position")
test_dynamixel.write_profile_velocity(80)
test_dynamixel.enable_torque()

# --- Input loop for position control ---
while True:
    try:
        theta = float(input("Enter dynamixel position ('q' to quit): "))
        test_dynamixel.write_position(theta)
    except ValueError:
        print("Exiting...")
        break

# --- End communication ---
test_dynamixel.end_communication()
