# RL-Drone-Stabilization

The aim of the project is to implement a Reinforcement Learning Architecture able to control aerial quadricopters, even in commercial drones.
The scenario concerns the advanced stabilization of a drone in critical situations such as high-speed wind, recovery due to a free fall, a propeller failure and any kind of features that could ensure safety during a flight in densely populated areas.

First of all we developed the project in a virtual environment, using the AirSim simulator, based on Unreal Engine, which was created precisely for the purpose of supporting the development of reinforcement learning applications for vehicles and quadcopters.

The idea was to initially simulate it in a virtual environment, to train the agent through deep RL, developing an MLP that runs with low latency.
In the future it would be interesting to try to deploy, safely, on a physical drone.


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

9. Add settings.json inside airsim folder (settings.json is a file containing all the quadricopter settings)

10. Open .sln with Visual Studio 2022 (or 2019 if the only installed), as suggested in this [link](https://docs.microsoft.com/it-it/visualstudio/ide/how-to-set-multiple-startup-projects?view=vs-2022) set Blocks as default Project, DebugGame Editor & Win64. Finally press F5 

11. Once Unreal is open with the project, click "Play" and use the keyboard to move the drone.

## Python Interface with AirSim

1. Take AirSim-1.7.0-windows/PythonClient/multirotor/hello_drone.py

2. Delete first line of import.

3. Create an Anaconda environment.

4. Install the following libraries
    ```bash
    pip install numpy
    pip install opencv-python
    pip install msgpack-rpc-python
    pip install airsim
    ```
5. Install Visual Studio & recommended python extensions (optional)

6. Unreal might lag if there is another window on top.To avoid this go in Unreal Engine settings: Edit->Editor preferences->search Performance->disable "Use less CPU when in background"

## Run the project
1. Clone the repository
    ```bash
    git clone https://github.com/lap98/RL-Drone-Stabilization.git
    ```
2. Open the environment in Unreal Engine

3. Run first.py in order to control the drone

## Reinforcement learning

In order to use TF-Agents library:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.9
pip install tf-agents==0.13.0
```
