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

# TF-agents library at the following link https://www.tensorflow.org/agents and the tutorial https://www.tensorflow.org/agents/tutorials/0_intro_rl


# How to setup the environment https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059
# py_environment https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment?hl=en#get_state
class DroneEnvironment(py_environment.PyEnvironment):

  def __init__(self):

    self.steps = 0

    # Action space with respective min and max values (control motors power)
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(4), dtype=np.float32, minimum=0, maximum=1, name='action')
    
    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)

    self.client.reset()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    self.client.takeoffAsync().join()

    self.initialPose = self.client.simGetObjectPose(object_name="SimpleFlight")

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

    # The state of the environment which can be seen by the drone using its sensors
    # The state represents also the input of the network
    self._state = [self.client.getImuData().angular_velocity.x_val, self.client.getImuData().angular_velocity.y_val, self.client.getImuData().angular_velocity.z_val, 
                  self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
                  self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6 ]
    self._episode_ended = False


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  # Reset the custom environment we created
  def _reset(self):
    self.steps = 0
    self._episode_ended = False
    self.client.reset()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    self.client.takeoffAsync().join()
    self.initialPose = self.client.simGetObjectPose(object_name="SimpleFlight")
    self._state = [self.client.getImuData().angular_velocity.x_val, self.client.getImuData().angular_velocity.y_val, self.client.getImuData().angular_velocity.z_val, 
                  self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
                  self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6 ]
    return ts.restart(np.array([self._state], dtype=np.float32))

  def move(self, action, duration = 0.1, continuous = False, step_treshold = 100):

    if self.client.simGetCollisionInfo().has_collided or self.steps > step_treshold:
      return True

    # Discrete actions
    if continuous == False:
      self.client.hoverAsync()
      
      self.client.moveByMotorPWMsAsync(front_right_pwm=action[0], rear_left_pwm=action[1], front_left_pwm=action[2], rear_right_pwm=action[3], duration=duration)
      if self.client.simGetCollisionInfo().has_collided:
        return True
  
      return False
    
    # Continuous Movements
    else:
      self.client.moveByMotorPWMsAsync(front_right_pwm=action[0], rear_left_pwm=action[1], front_left_pwm=action[2], rear_right_pwm=action[3], duration=20)
      if self.client.simGetCollisionInfo().has_collided:
          return True

  def getState(self):
    return [self.client.getImuData().angular_velocity.x_val, self.client.getImuData().angular_velocity.y_val, self.client.getImuData().angular_velocity.z_val, 
                  self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
                  self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6 ]


  def reward_function(self,pose):
    reward = 1/(np.sqrt((pose.position.x_val - self.initialPose.position.x_val)**2+(pose.position.y_val - self.initialPose.position.y_val)**2+(pose.position.z_val - self.initialPose.position.z_val)**2))
    reward = reward + 1/(np.sqrt((pose.orientation.w_val - self.initialPose.orientation.w_val)**2+(pose.orientation.x_val - self.initialPose.orientation.x_val)**2+(pose.orientation.y_val - self.initialPose.orientation.y_val)**2+(pose.orientation.z_val - self.initialPose.orientation.z_val)**2))
    return reward

  def _step(self, action):

    # If the episode is done, reset the environment
    if self._episode_ended:
      return self.reset()
    # Perform the chosen move (if collision occurs, it returns True)
    self._episode_ended = self.move(action = action)
    # Get the new state
    self._state = self.getState()
    # Determine reward
    reward = self.reward_function(self.client.simGetObjectPose(object_name="SimpleFlight"))
    # Count steps
    self.steps += 1
    # Call episode end if person is out of view for too long or reach 
    # our limit
    if self.steps >= 50:
      self._episode_ended = True
      self.time_end = True
    # Handle terminal states
    if self._episode_ended:
      # If collision occured, we give a negative reward
      if self.time_end == False:
          reward = -1
    # Return a terminal state to the agent
      return ts.termination(np.array(self._state, dtype=np.float32),
                            reward)
    # If were still going on we transition to the next state
    else:
      return ts.transition(np.array(self._state, dtype=np.float32), 
                            reward=reward, 
                            discount=0.9)