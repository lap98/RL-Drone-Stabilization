import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

wind = airsim.Vector3r(0,0,0)
client.simSetWind(wind)

airsim.wait_key('Press any key to disarm')
print("Disarmed...")
client.armDisarm(True)
client.takeoffAsync().join()

wind = airsim.Vector3r(50,0,0)
client.simSetWind(wind)
time.sleep(30)
client.moveByMotorPWMsAsync(1, 1, 1, 1, 5)
airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(True)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
