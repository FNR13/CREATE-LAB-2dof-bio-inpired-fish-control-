# ============================
# Kinematics Configuration
# ============================

# Gear & pulley geometry
GEAR_RATIO = 21.0 / 26.0     # servo gear ratio
R_SERVO_PULLEY = 7.25        # radius of servo pulley (mm)
R_FIN_PULLEY  = 17.5         # radius of fin pulley (mm)

# Polynomial coefficients for:
# C4*θ⁴ + C3*θ³ + C2*θ² + C1*θ + C0
C4 = 5.77e-08      # θ⁴ term
C3 = -4.303e-05    # θ³ term
C2 = -2.631e-05    # θ² term
C1 = 1.045         # θ¹ term
C0 = 0.2932        # constant term

# Dynamixel calibration values
DYNAMIXEL_90_VALUE = 570
DYNAMIXEL_0_VALUE = 1600
# ============================
import math
import numpy as np

def fin_to_servo(phi):
    """
    Convert fin , in degrees to servo angle, in degrees
    using pulley and gear ratios.
        f(phi_FIN) --> theta_SERVO
    """
    k = GEAR_RATIO * (R_SERVO_PULLEY / R_FIN_PULLEY)  # transmission ratio

    theta = phi / k + 90.0            # servo neutral at 90°

    # theta = max(0.0, min(theta, 180.0))  # clamp between 0 and 180°
    theta = np.clip(theta, 0.0, 180.0) 

    return theta


def tail_f(theta):
    """
    Tail kinematics function 
        f(theta_DYNAMIXEL) --> phi_TAIL
    """
    return(
        C4 * theta**4 +
        C3 * theta**3 +
        C2 * theta**2 +
        C1 * theta    +
        C0
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
    -90° → DYNAMIXEL_-90_VALUE
     0°  → DYNAMIXEL_0_VALUE
    +90° → DYNAMIXEL_90_VALUE
    """
    slope = -(DYNAMIXEL_0_VALUE-DYNAMIXEL_90_VALUE)/90
    offset = DYNAMIXEL_0_VALUE
    val = slope * angle_deg + offset
    return int(round(val))

def dynamixel_position_to_angle(angle_pos):
    """
    Convert position to Dynamixel angle in degrees.
    Based on calibration:
    -90° → DYNAMIXEL_-90_VALUE
     0°  → DYNAMIXEL_0_VALUE
    +90° → DYNAMIXEL_90_VALUE
    """
    slope = -90/(DYNAMIXEL_0_VALUE-DYNAMIXEL_90_VALUE)
    offset = DYNAMIXEL_0_VALUE
    val = slope * (angle_pos - offset)
    return int(round(val))
