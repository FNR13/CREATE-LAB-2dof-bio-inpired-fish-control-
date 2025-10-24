import sys
import time

# MAKE SURE TO CLOSE DYNAMIXEL WIZARD AND ARDUINO IDE BEFORE RUNNING THIS TEST
# END COMUNNICATIONS TO AVOID ISSUES IN RERUNNING THE TEST

# adding Folder_2 to the system path
sys.path.insert(0, 'support_scripts_py')

from fish_control_comms import Fish_Control_Comms   
from kinematics import inverse_tail, dynamixel_angle_to_position


fish_robot = Fish_Control_Comms(
    arduino_port="COM17", arduino_baudrate=115200, 
    dynamixel_port="COM18", dynamixel_baudrate=57600, dynamixel_ID=1, dynamixel_velocity=80
)

fish_robot.connect_devices()

def main():   
    print("Starting Fish Control Comms Test")

    # for i in range(5):
    #     fish_robot.set_PWM_Angle(90+2*i)
    #     print(f"Set PWM Angle to {90+i}")
    #     time.sleep(0.5)

    print("\n")
    print("Moving Dynamixel to 0 degrees")
    value = dynamixel_angle_to_position((0))
    print("Dynamixel value:", value)
    fish_robot.move_dynamixel(value)

    input("Press Enter to continue...")

    print("\n")
    print("Moving Dynamixel to 78 degrees")
    value = dynamixel_angle_to_position((78))
    print("Dynamixel value:", value)
    fish_robot.move_dynamixel(value)

    input("Press Enter to continue...")

    print("\n")
    print("Moving Dynamixel to -78 degrees")
    value = dynamixel_angle_to_position((-78))
    print("Dynamixel value:", value)
    fish_robot.move_dynamixel(value)
    
if __name__ == "__main__":
    main()