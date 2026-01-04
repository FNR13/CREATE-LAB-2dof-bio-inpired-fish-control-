class PID_Controller:
    """
    Simple standalone PID controller.
    """

    def __init__(self, Kp, Ki=0.0, Kd=0.0, output_limits=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0

        self.output_limits = output_limits

    def update(self, reference, state, dt):
        
        error = reference - state
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if self.output_limits:
            min_out, max_out = self.output_limits
            output = max(min(output, max_out), min_out)

        return output

    def reset(self):
        """Reset the integral and derivative terms"""
        self.integral = 0.0
        self.prev_error = 0.0
