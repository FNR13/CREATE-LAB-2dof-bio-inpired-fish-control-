from controllers.pid_controller import PID_Controller

class SimpleController:
    """Controller that uses two PID controllers: one for surge and one for yaw."""

    def __init__(self,
                 surge_gains=(1.0, 0.0, 0.0),
                 yaw_gains=(1.0, 0.0, 0.0),
                 amplitude_limits=(5, 45),       # degrees
                 bias_limits=(-15, 15)):         # degrees

        Kp_s, Ki_s, Kd_s = surge_gains
        Kp_y, Ki_y, Kd_y = yaw_gains

        # Important: PID limits should be (min, max)
        self.surge_controller = PID_Controller(Kp_s, Ki_s, Kd_s,
                                               output_limits=amplitude_limits)
        self.yaw_controller   = PID_Controller(Kp_y, Ki_y, Kd_y,
                                               output_limits=bias_limits)

    def update(self, surge_ref, surge_state, yaw_ref, yaw_state, dt):
        
        amplitude = self.surge_controller.update(surge_ref, surge_state, dt)
        bias      = self.yaw_controller.update(yaw_ref, yaw_state, dt)

        return amplitude, bias

    def reset(self):
        self.surge_controller.reset()
        self.yaw_controller.reset()
