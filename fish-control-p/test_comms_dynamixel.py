import sys
import time

import numpy as np

# ---------------------------------------------------------------------
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMUNNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST
# ---------------------------------------------------------------------

# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from dynamixel_controller import Dynamixel

test_dynamixel = Dynamixel(ID=1, descriptive_device_name="Flag dynamixel", port_name="COM18", baudrate=57600)

test_dynamixel.begin_communication()
test_dynamixel.set_operating_mode("position")
test_dynamixel.write_profile_velocity(80)
test_dynamixel.enable_torque()


while True:
    try:
        theta = float(input("Enter ynamixel position ('q' to quit): "))
        test_dynamixel.write_position(theta)
    except ValueError:
        print("Exiting...")
        break

input("Press Enter to end comunication...")
test_dynamixel.end_communication()