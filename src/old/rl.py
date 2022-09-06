# Import libraries
import airsim

# Connect to the AirSim Drone
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync()

# Reset method from TF-Agents must be overwritten

# Also the step method must be overwritten

front_right_pwm = 0.0
rear_left_pwm = 0.0
front_left_pwm = 0.0
rear_right_pwm = 0.0
duration = 10
client.moveByMotorPWMsAsync(1, 1, 1, 1, duration)
