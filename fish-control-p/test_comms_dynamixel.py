import sys
import time

# MAKE SURE TO CLOSE DYNAMIXEL WIZARD AND ARDUINO IDE BEFORE RUNNING THIS TEST
# END COMUNNICATIONS TO AVOID ISSUES IN RERUNNING THE TEST

# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from dynamixel_controller import Dynamixel
from kinematics import inverse_tail

test_dynamixel = Dynamixel(ID=1, descriptive_device_name="Flag dynamixel", port_name="COM18", baudrate=57600)

test_dynamixel.begin_communication()
test_dynamixel.set_operating_mode("position")
test_dynamixel.write_profile_velocity(80)
test_dynamixel.enable_torque()


test_dynamixel.write_position(inverse_tail((2.5)))

input("Press Enter to end comunication...")
test_dynamixel.end_communication()