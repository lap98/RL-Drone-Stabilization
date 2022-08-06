#https://microsoft.github.io/AirSim/api_docs/html/
import airsim
import pprint
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

dist_data = client.getDistanceSensorData(vehicle_name="SimpleFlight")
print(f"Distance sensor data: {dist_data.distance}")

client.takeoffAsync().join()
client.moveByMotorPWMsAsync(front_right_pwm=1, rear_left_pwm=1, front_left_pwm=1, rear_right_pwm=1, duration=5).join()
#client.moveByMotorPWMsAsync(front_right_pwm=0, rear_left_pwm=0, front_left_pwm=0.1, rear_right_pwm=0.1, duration=5)
#client.cancelLastTask()

dist_data = client.getDistanceSensorData(vehicle_name="SimpleFlight")
print(f"Distance sensor data: {dist_data.distance}")

# let's quit cleanly
#client.armDisarm(False)
#client.enableApiControl(False)

# LOSS FUNCTION 
# Orientation and Position
q = client.simGetVehiclePose()
print(q)

# RESET
 # Initialize the drone position

# STEP
# 


class CardGameEnv(py_environment.PyEnvironment):

  def __init__(self):
    # Action space with respective min and max values (control motors power)
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(4), dtype=np.float32, minimum=0, maximum=1, name='action')

    # Observation space 
    # angular_velocity.x_val
    # angular_velocity.y_val
    # angular_velocity.z_val
    # linear_acceleration.x_val
    # linear_acceleration.y_val
    # linear_acceleration.z_val
    # dist_data.distance (distance from the ground)
    # barometro
    self._observation_spec = array_spec.ArraySpec(
        shape=(8,), dtype=np.float32, name='observation')

    # The state contains everything required to restore 
    # the environment to the current configuration.
    # To check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    self._state = 0
    self._episode_ended = False


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec


  def _reset(self):
    self._episode_ended = False
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    self._state = 0
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)