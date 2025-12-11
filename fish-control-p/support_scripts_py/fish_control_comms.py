try:
    from .dynamixel_controller import Dynamixel
    from .comms_wrapper import Arduino
except:
    from dynamixel_controller import Dynamixel
    from comms_wrapper import Arduino

class Fish_Control_Comms():
    def __init__(self, arduino_port, arduino_baudrate, dynamixel_port, dynamixel_baudrate, dynamixel_ID, dynamixel_velocity) -> None:
        self.dynamixel = Dynamixel(ID=dynamixel_ID, descriptive_device_name="Flag dynamixel", port_name=dynamixel_port, baudrate=dynamixel_baudrate)
        self.arduino = Arduino(descriptiveDeviceName="Flag arduino", portName=arduino_port, baudrate = arduino_baudrate)
        self.dynamixel_velocity = dynamixel_velocity

    def connect_devices(self):
        self.arduino.connect_and_handshake()
        self.arduino.send_message("halt")

        self.dynamixel.begin_communication()
        self.dynamixel.set_operating_mode("position")
        self.dynamixel.write_profile_velocity(self.dynamixel_velocity)
        self.dynamixel.enable_torque()

    def set_PWM_Angle(self, angle):
        self.arduino.send_message(f"pwm:{angle}")

    def move_dynamixel(self, position):
        self.dynamixel.write_position(position)

    def get_dynamixel_position(self):
        return self.dynamixel.read_position()
    
    def disconnect_devices(self):
        self.dynamixel.end_communication()

   