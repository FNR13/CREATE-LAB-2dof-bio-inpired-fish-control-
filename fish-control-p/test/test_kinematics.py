# Add support folder to path
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
support_dir = os.path.join(parent_dir, 'support_scripts_py')

sys.path.insert(0, support_dir)

import numpy as np
import matplotlib.pyplot as plt

from kinematics import fin_to_servo, inverse_tail, tail_f 

def main():
    print("=== Kinematics Test ===")
    print()

    # Test 1: Fin to Servo Mapping
    test_phis = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
    print("Fin-to-Servo mapping:")
    for phi in test_phis:
        theta = fin_to_servo(phi)
        print(f"  Fin deflection {phi:>5.1f}° → Servo angle {theta:6.2f}°")
    print()

    # Test 2: Inverse Tail Mapping
    test_tails = [-20, -10, 0, 10, 20]
    print("Inverse Tail mapping:")
    for phi in test_tails:
        theta = inverse_tail(phi)
        print(f"  Tail deflection {phi:>5.1f}° → Dynamixel {theta:6.2f}")
    print()


    # Generate input range for phi (in degrees or radians — your choice)
    phi = np.linspace(-90, 90, 200)  # adjust range as needed

    # Compute output using your function
    theta = fin_to_servo(phi)

    # Plot
    plt.figure()
    plt.plot(phi, theta)
    plt.title("fin_to_servo Relationship")
    plt.xlabel("Phi (input)")
    plt.ylabel("Theta (output)")
    plt.grid(True)
    plt.show()
    # # Optional: interactive test
    # while True:
    #     try:
    #         phi = float(input("Enter fin deflection (degrees, 'q' to quit): "))
    #         print(f"Servo angle = {fin_to_servo(phi):.2f}°")
    #     except ValueError:
    #         print("Exiting...")
    #         break


if __name__ == "__main__":
    main()