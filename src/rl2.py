#https://microsoft.github.io/AirSim/api_docs/html/
import airsim
import pprint

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
#client.reset()

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

dist_data = client.getDistanceSensorData(vehicle_name="SimpleFlight")
print(f"Distance sensor data: {dist_data.distance}")

#client.takeoffAsync().join()
client.moveByMotorPWMsAsync(front_right_pwm=1, rear_left_pwm=1, front_left_pwm=1, rear_right_pwm=1, duration=5).join()
#client.moveByMotorPWMsAsync(front_right_pwm=0, rear_left_pwm=0, front_left_pwm=0.1, rear_right_pwm=0.1, duration=5)
#client.cancelLastTask()

dist_data = client.getDistanceSensorData(vehicle_name="SimpleFlight")
print(f"Distance sensor data: {dist_data.distance}")

# let's quit cleanly
#client.armDisarm(False)
#client.enableApiControl(False)