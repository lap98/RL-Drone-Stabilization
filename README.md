# RL-Drone-Stabilization

## Environmental Setup
1. Install Unreal Engine (4.27 suggested) from [Epic Games Launcher](https://store.epicgames.com/it/download).

2. Install Visual Studio 2019

3. Install C++ dev

4. Install Python

5. Download [AirSim](https://microsoft.github.io/AirSim/build_windows/) prebuilt source code and the environment of your choice.

6. Place the Environment in AirSim/Unreal/Environment

5. Use Visual Studio 2019 Developer Command Prompt with Admin privileges to run AirSim-1.7.0-windows/build.cmd

6. Follow the [tutorial](https://microsoft.github.io/AirSim/unreal_blocks/) in order to setup Blocks Environment for AirSim

7. Install [.net framework](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net462) 4.6.2 Developer (SDK), desktop runtime 3.1.24 

8. Run AirSim-1.7.0-windows/Unreal/Environments/Blocks/update_from_git.bat

9. Open .sln with Visual Studio 2022 (or 2019 if the only installed), as suggested in this [link](https://docs.microsoft.com/it-it/visualstudio/ide/how-to-set-multiple-startup-projects?view=vs-2022) set Blocks as default Project, DebugGame Editor & Win64. Finally press F5 

8. Once Unreal is open with the project, click "Play" and use the keyboard to move the drone.

## Python Interface with AirSim

0. Take AirSim-1.7.0-windows/PythonClient/multirotor/hello_drone.py

1. Delete first line of import.

2. Create an Anaconda environment.

3. Install the following libraries
    ```bash
    pip install numpy
    ```
    ```bash
    pip install opencv-python
    ```
    ```bash
    pip install msgpack-rpc-python
    ```
    ```bash
    pip install airsim
    ```

## Run the project
1. Clone the repository
    ```bash
    git clone https://github.com/lap98/RL-Drone-Stabilization.git
    ```
2. Open the environment in Unreal Engine

3. Run first.py in order to control the drone

