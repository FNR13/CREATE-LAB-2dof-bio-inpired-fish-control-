import sys
import time

# MAKE SURE TO CLOSE ARDUINO IDE BEFORE RUNNING THIS TEST
# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from comms_wrapper import Arduino

arduino_test = Arduino(descriptiveDeviceName="Flag arduino", portName="COM17", baudrate = 115200)

arduino_test.connect_and_handshake()

time.sleep(2)  # wait for connection to establish

arduino_test.send_message(f"pwm:{100}")

# Finish Communication
arduino_test.send_message("halt")
arduino_test._disect_and_save_message()