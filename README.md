# AUTHOR : Ricardo Francisco 
# SEMESTER PROJECT : BIOMIMETIC FISH OSCILLATORY Control

# 🐟 Fish Robot Controller

A control system for a robotic fish with synchronized tail and fin movements. The system consists of a Dynamixel servo motor for the tail and an Arduino-controlled servo for the fin.

## Hardware Setup

- **Tail**: Dynamixel servo motor (ID: 1, 57600 baud)
- **Fin**: Standard servo motor controlled via Arduino (115200 baud)

## Software Components

### Arduino
- `flag_arduino.ino` - Main Arduino sketch
- `pyCommsLib` - Communication protocol library

### Python Control
Main test scripts:
- `water_tests.py` - Main control loop for water testing
- `test_loop.py` - Simulated control loop (no hardware)
- `test_kinematics.py` - Test kinematics transformations
- `test_comms.py` - Test full communication system

Support modules in `support_scripts_py`:
- `fish_control_comms.py` - High-level control interface  
- `kinematics.py` - Fin/tail angle transformations
- `dynamixel_controller.py` - Dynamixel motor interface
- `comms_wrapper.py` - Serial communication wrapper


## Installation

1. Create a conda environment with th e provided `.yml`:
```sh
conda env create --name envname --file=environment.yaml
conda activate envname
```

2. Upload `flag_arduino.ino` to Arduino board (make sure it has the C++ and header files)

## Usage

1. Connect hardware:
   - Arduino (115200 baud)
   - Dynamixel (57600 baud)

2. Upload `flag_arduino` to arduino

3. Run python code

