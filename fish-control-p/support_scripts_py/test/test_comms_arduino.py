import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import time
from comms_wrapper import Arduino

# --- Warning / instructions ---
# MAKE SURE TO CLOSE ARDUINO IDE AND DYNAMIXEL WIZARD (or disconnect) BEFORE RUNNING THIS TEST
# END COMMUNICATIONS IN THE END TO AVOID ISSUES IN RERUNNING THE TEST


# --- Connect to Arduino ---
arduino_test = Arduino(descriptiveDeviceName="Flag arduino", portName="COM17", baudrate=115200)
arduino_test.connect_and_handshake()
time.sleep(2)  # wait for connection to establish

# --- Mode selection ---
mode = "input"
# mode = "sweep"

# --- Sweep mode ---
if mode == "sweep":
    arduino_test.send_message("mode:sweep")
    delay = 0.020

    for i in range(30):
        arduino_test.send_message(f"pwm:{90 + i}")
        time.sleep(delay)

    while True:
        for i in range(60):
            arduino_test.send_message(f"pwm:{120 - i}")
            time.sleep(delay)

        for i in range(60):
            arduino_test.send_message(f"pwm:{60 + i}")
            time.sleep(delay)

# --- Input mode ---
if mode == "input":
    while True:
        try:
            theta = float(input("Enter servo position ('q' to quit): "))
            arduino_test.send_message(f"pwm:{theta}")
        except ValueError:
            print("Exiting...")
            break

# --- Finish communication ---
arduino_test.send_message("halt")
