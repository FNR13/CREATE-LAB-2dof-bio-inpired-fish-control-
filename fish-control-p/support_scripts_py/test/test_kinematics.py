# Imports
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from kinematics import fin_to_servo, inverse_tail, tail_f

print("=== Kinematics Test ===\n")

# --- Test 1: Fin to Servo Mapping ---
test_phis = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
print("Fin-to-Servo mapping:")
for phi in test_phis:
    theta = fin_to_servo(phi)
    print(f"  Fin deflection {phi:>5.1f}° → Servo angle {theta:6.2f}°")
print()

# --- Test 2: Inverse Tail Mapping ---
test_tails = [-20, -10, 0, 10, 20]
print("Inverse Tail mapping:")
for phi in test_tails:
    theta = inverse_tail(phi)
    print(f"  Tail deflection {phi:>5.1f}° → Dynamixel {theta:6.2f}")
print()

# --- Generate plot for fin_to_servo ---
phi_range = np.linspace(-90, 90, 200)  # adjust range as needed
theta_range = fin_to_servo(phi_range)

plt.figure()
plt.plot(phi_range, theta_range)
plt.title("fin_to_servo Relationship")
plt.xlabel("Phi (input)")
plt.ylabel("Theta (output)")
plt.grid(True)
plt.show()

# --- Optional interactive test ---
# while True:
#     try:
#         phi = float(input("Enter fin deflection (degrees, 'q' to quit): "))
#         print(f"Servo angle = {fin_to_servo(phi):.2f}°")
#     except ValueError:
#         print("Exiting...")
#         break
