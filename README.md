# AUTHOR: Ricardo Francisco  
# SEMESTER PROJECT: Fish Control Framework

# 🐟 Fish Robot Controller

A control system for a robotic fish with synchronized tail and fin movements. The system uses a Dynamixel servo motor for the tail and an Arduino-controlled servo motor for the caudal fin.

## Hardware Setup

To assemble the fish hardware, refer to `Hardware_Setup.png` for better visibility.

To connect the robotic fish:
- Connect the U2D2 adapter with the power hub to the fish.
- Connect the power hub to the power supply.
- Connect the U2D2 adapter to the laptop via USB.
- Connect the laptop to the Arduino via USB (ensure the correct firmware is uploaded).
- Connect Arduino pin 6 (as defined in the code) to the servo signal line (white cable).
- Connect the Arduino ground to the ground provided by the power supply.
- Connect the +12~V power line to the fish cable.

Ensure that the correct COM ports for both the Arduino and the Dynamixel motor are set in the code.

For pool experiments, simply connect the camera to the laptop via USB.

All cables and boards are should be stored in the drawer labeled **"GAB"**.

Motor Specifications (they are already set in code):

- **Tail**: Dynamixel servo motor (ID: 1, 57600 baud)
- **Fin**: Standard servo motor controlled via Arduino (115200 baud)

## Software Components

### Arduino

Upload the following files to the Arduino:
- `flag_arduino.ino` — Main Arduino sketch
- `pyCommsLib` — Communication protocol library

### Python Control

GitRepo/
├── environment.yaml           # Python environment file with dependencies
├── flag_arduino/
│   ├── flag_arduino.ino       # Arduino main code
│   ├── pyCommsLib.cpp         # Communication support library
│   └── pyCommsLib.h
├── python/                    # Python source code
│   ├── examples/              # Scripts for open-loop and closed-loop tests
│   ├── fish_controller/    
│   ├── support_scripts_py/    # Motor command and utility scripts
│   └── vision/          
│        ├── media/
│        ├── test/
│        ├── vision_supp/      # Debug and testing utilities
│        ├── vision_config.py
│        ├── vision_helpers.py
│        └── vision.py         # Complete vision pipeline implementation
└── reports/                   # Project documentation
   ├── Gabriel_thesis.pdf
   ├── Ricardo_Project
   └── UserManual/             # Instructions for fish assembly and disassembly


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

