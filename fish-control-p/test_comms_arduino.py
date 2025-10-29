import sys
import time

# ---------------------------------------------------------------------
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMUNNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST
# ---------------------------------------------------------------------

# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from comms_wrapper import Arduino

arduino_test = Arduino(descriptiveDeviceName="Flag arduino", portName="COM17", baudrate = 115200)

arduino_test.connect_and_handshake()

time.sleep(2)  # wait for connection to establish

mode = "input"
mode = "sweep"


if mode == "sweep":
    arduino_test.send_message("mode:sweep")
    delay = 0.020
    for i in range(30):
        arduino_test.send_message(f"pwm:{90+ i}")
        time.sleep(delay)

    while True:
        for i in range(60):
            arduino_test.send_message(f"pwm:{120 - i}")
            time.sleep(delay)

        for i in range(60):
            arduino_test.send_message(f"pwm:{60 + i}")
            time.sleep(delay)

if mode == "input":
    while True:
        try:
            theta = float(input("Enter servo position ('q' to quit): "))
            arduino_test.send_message(f"pwm:{70}")
        except ValueError:
            print("Exiting...")
            break

# Finish Communication
arduino_test.send_message("halt")