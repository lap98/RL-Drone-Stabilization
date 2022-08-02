# Import libraries
import airsim

# Connect to the AirSim Drone
client = airsim.MultirotorClient()
client.confirmConnection()
airsim.DrivetrainType.ForwardOnly
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Reset method from TF-Agents must be overwritten

# Also the step method must be overwritten
front_right_pwm = 1
rear_left_pwm = 1
front_left_pwm = 1
rear_right_pwm = 1
duration = 5
client.moveByMotorPWMsAsync(front_right_pwm, rear_left_pwm, front_left_pwm, rear_right_pwm, duration)