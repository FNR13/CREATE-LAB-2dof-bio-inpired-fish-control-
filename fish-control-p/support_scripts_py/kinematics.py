import math

def fin_to_servo(phi):
    """
    Convert fin , in degrees to servo angle, in degrees
    using pulley and gear ratios.
        f(phi_FIN) --> theta_SERVO
    """
    gear_ratio = 21.0 / 26.0          # gear ratio (servo gear)
    r_servo_pulley = 7.25             # radius of servo pulley (mm)
    r_fin_pulley = 17.5               # radius of fin pulley (mm)
    k = gear_ratio * (r_servo_pulley / r_fin_pulley)  # transmission ratio

    theta = phi / k + 90.0            # servo neutral at 90°

    theta = max(0.0, min(theta, 180.0))  # clamp between 0 and 180°

    return theta


def tail_f(theta):
    """
    Tail kinematics function 
        f(theta_DYNAMIXEL) --> phi_TAIL
    """
    return(
        5.77e-08    * theta**4
        - 4.303e-05 *theta**3
        - 2.631e-05 *theta**2
        + 1.045     *theta 
        + 0.2932
    )


def inverse_tail(phi):
    """
    Inverse tail kinematics: given tail deflection phi, find dynamixel angle theta.
    Uses bisection search on f(theta).
        f(phi_TAIL) --> theta_DYNAMIXEL
    """
    sign = -1.0 if phi < 0 else 1.0
    phi = abs(phi)

    low, high = 0.0, 90.0
    if phi < tail_f(low):
        return 0.0
    if phi > tail_f(high):
        return sign * high

    mid = 0.0
    for _ in range(30):
        mid = 0.5 * (low + high)
        fm = tail_f(mid)
        if abs(fm - phi) < 0.01:
            break
        if fm < phi:
            low = mid
        else:
            high = mid

    return sign * mid

def dynamixel_angle_to_position(angle_deg):
    """
    Convert gear angle in degrees to Dynamixel position value.
    Based on calibration:
    -90° → 2630
     0°  → 1600
    +90° → 570
    """
    slope = -(1600-570)/90
    offset = 1600
    val = slope * angle_deg + offset
    return int(round(val))

def dynamixel_position_to_angle(angle_pos):
    """
    Convert position to Dynamixel angle in degrees.
    Based on calibration:
    -90° → 2630
     0°  → 1600
    +90° → 570
    """
    slope = -90/(1600-570)
    offset = 1600
    val = slope * (angle_pos - offset)
    return int(round(val))
