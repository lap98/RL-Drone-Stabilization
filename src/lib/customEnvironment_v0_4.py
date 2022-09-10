# AirSim docs: https://microsoft.github.io/AirSim/api_docs/html/
# TF-agents library at the following link https://www.tensorflow.org/agents and the tutorial https://www.tensorflow.org/agents/tutorials/0_intro_rl
# How to setup the environment https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059
# py_environment https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment?hl=en#get_state
# https://towardsdatascience.com/cartpole-problem-using-tf-agents-build-your-first-reinforcement-learning-application-3e6006adeba7

#################################################
# Imports
#################################################

import os
import time
import airsim
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec


#################################################
# AirSim environment definition
#################################################

class DroneEnvironment(py_environment.PyEnvironment):

  '''Initializes the environment, connecting to the drone and setting the observation and action spaces
  :param enable_wind: whether to enable the wind
  :param randomize_initial_pose: whether to randomize the initial position and orientation of the drone
  '''
  def __init__(self, enable_wind=False, randomize_initial_pose=False):
    self.enable_wind = enable_wind
    self.randomize_initial_pose = randomize_initial_pose

    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    # Observation space: IMU (3x angular velocity, 3x linear acceleration), ultrasound distance from the ground, barometer
    self._observation_spec = array_spec.ArraySpec(shape=(13,),dtype=np.float32, name='observation')
    # Action space: control motors power
    self._action_spec = array_spec.BoundedArraySpec(shape=(5,), dtype=np.float32, name='action', minimum=0.0, maximum=1.0)
    # The state of the environment which can be seen by the drone using its sensors; it represents also the input of the network
    #self._state = self.getState()
    self._episode_ended = False
    self._total_reward = 0
  
  def action_spec(self):
    return self._action_spec
  
  def observation_spec(self):
    return self._observation_spec

  '''Generates a new pose, randomized or not
  '''
  def getNewPose(self, random=False, random_uniform=False):
    pos_stddev = 0.5 # in [m]
    or_stddev = 0.15 # in [rad], 0.15 -> at most around 20 deg of inclination
    if random:
      if random_uniform:
        u = np.random.uniform()
        v = np.random.uniform()
        w = np.random.uniform()
        new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0 + np.random.normal(0, pos_stddev), 0.0 + np.random.normal(0, pos_stddev), -100.0 + np.random.normal(0, pos_stddev)),
                                orientation_val=airsim.Quaternionr(np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v), np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)))
      else:
        new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0 + np.random.normal(0, pos_stddev), 0.0 + np.random.normal(0, pos_stddev), -100.0 + np.random.normal(0, pos_stddev)),
                                orientation_val=airsim.utils.to_quaternion(np.random.normal(0, or_stddev), np.random.normal(0, or_stddev), np.random.normal(0, or_stddev))) # roll pitch yaw in radians, to quaternion
    else:
      new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0, 0.0, -100.0), orientation_val=airsim.Quaternionr(0.0, 0.0, 0.0, 1.0))
    reference_pose = airsim.Pose(position_val=airsim.Vector3r(0.0, 0.0, -100.0), orientation_val=airsim.Quaternionr(0.0, 0.0, 0.0, 1.0))
    return new_pose, reference_pose
  
  '''Resets the pose of the drone to what is specified inside the function and prints the state of the multirotor (flying or not)
  '''
  def reset_pose(self):
    self.client.reset()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    self.client.takeoffAsync(timeout_sec=0.1, vehicle_name="SimpleFlight").join() # needed to enable the multirotor correct state

    new_pose, self.initial_pose = self.getNewPose(self.randomize_initial_pose)

    #self.client.simSetObjectPose(object_name="SimpleFlight", pose=new_pose, teleport=False)
    self.client.simSetVehiclePose(pose=new_pose, ignore_collision=False, vehicle_name="SimpleFlight")
    if self.client.getMultirotorState().landed_state == airsim.LandedState.Landed: print("LANDED: Physics Engine NOT Engaged")
    else: print("[CORRECTLY FLYING: Physics Engine Engaged]")
    time.sleep(0.5) # needed because otherwise, the neural net is too fast and the _steps begin too soon, when the drone is not ready, and no experience would be gained (rew=0)
  
  '''Sets a random wind in the simulation, with the given standard deviation in [m]
  '''
  def setRandomWind(self, stddev=5):
    x_val = np.random.normal(0, stddev)
    y_val = np.random.normal(0, stddev)
    z_val = np.random.normal(0, stddev)
    wind = airsim.Vector3r(x_val, y_val, z_val)
    print('Wind set < x y z >: <', x_val, y_val, z_val, '>')
    self.client.simSetWind(wind)
  
  '''Resets the custom environment created
  '''
  def _reset(self):
    print('Total reward for the previous episode:', self._total_reward)
    self._total_reward = 0
    if self.enable_wind: self.setRandomWind()
    
    self.reset_pose()

    self._state = self.getState()
    self._episode_ended = False
    return ts.restart(self._state)

  '''Override, one step: performs an action, retrieves the reward, returns either the transition or the termination signals associated
  :param action: the action to perform decided by the agent
  :return: either a transition or a termination
  '''
  def _step(self, action):
    # If the episode is done, reset the environment
    if self._episode_ended: return self.reset()

    # Perform the chosen move (if collision occurs, it returns True), get new state, compute reward
    end_now = self.move(action=action)
    self._state = self.getState()
    reward = self.reward_function(self.client.simGetVehiclePose(vehicle_name="SimpleFlight"))

    # Handle states - step_type: 0->beginning, 1->normal transition, 2->terminal
    if end_now:
      print('Collision occurred or episode termination condition met')
      self._episode_ended = True
      reward = 0 # a collision occured, give a NEGATIVE? reward
      return ts.termination(self._state, reward) # return terminal state to the agent
    else: # if were still going on we transition to the next state
      self._total_reward += reward
      return ts.transition(self._state, reward=reward)

  '''Moves the drone as specified by the action, first checking for the termination conditions
  :param action: the tensorflow action, as described by the array spec in the __init__ function
  :param duration: how long the duration of the action has to be. If the movemetnt is continuous, only specifies the maximum duration (it is asyncronous)
  :param continuous: whether to do continuous movements or wait for the action to end before going back to the network inference
  :return: True if the episode has to end due to collisions or other, False otherwise
  '''
  def move(self, action, duration=0.002, continuous=False):
    #if self.client.simGetCollisionInfo().has_collided or self.client.simGetVehiclePose().position.z_val > -10: return True
    #pose = self.client.simGetVehiclePose() # check if vehicle is too distant from original point -> collect more dense samples where it is needed
    #if self.client.simGetCollisionInfo().has_collided or abs(pose.position.z_val-self.initial_pose.position.z_val) > 5 or abs(pose.position.x_val-self.initial_pose.position.x_val) > 5 or abs(pose.position.y_val-self.initial_pose.position.y_val) > 5: return True

    if continuous == True: # continuous movements -> the control loop is: network inference -> perform action asynchronously -> ...
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(action[0]), rear_left_pwm=float(action[1]), front_left_pwm=float(action[2]), rear_right_pwm=float(action[3]), duration=duration)
    else: # discrete movements -> the control loop is: network inference -> perform action and join -> ...
      #self.client.hoverAsync()
      scale = 1.0 # scale for the delta thrust of each motor, how much importance to give it
      d_th = scale * (action[1:]-0.5) / 5 # delta thrust on the rotors
      th = np.clip(float(action[0]) + d_th, 0, 1) # thrust bias + delta, clipped
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(th[0]), rear_left_pwm=float(th[1]), front_left_pwm=float(th[2]), rear_right_pwm=float(th[3]), duration=duration).join()
      #self.client.moveByMotorPWMsAsync(front_right_pwm=float(0.5), rear_left_pwm=float(0.5), front_left_pwm=float(0.5), rear_right_pwm=float(0.5), duration=duration).join()
    
    return False

  '''Returns the state as a numpy array with float32 values
  '''
  def getState(self):
    pose = self.client.simGetVehiclePose()
    imu_data = self.client.getImuData()
    return np.array([(pose.position.z_val-self.initial_pose.position.z_val)/10, (pose.position.x_val-self.initial_pose.position.x_val)/10, (pose.position.y_val-self.initial_pose.position.y_val)/10, # between -1 and 1 more or less
                    imu_data.angular_velocity.x_val/10, imu_data.angular_velocity.y_val/10, imu_data.angular_velocity.z_val/10, # at most around 30ish?
                    imu_data.linear_acceleration.x_val/10, imu_data.linear_acceleration.y_val/10, imu_data.linear_acceleration.z_val/10, # at most around 30ish?
                    pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val], # between -1 and 1
                    dtype=np.float32)
    
    #return np.array([self.client.getImuData().angular_velocity.x_val/10, self.client.getImuData().angular_velocity.y_val/10, self.client.getImuData().angular_velocity.z_val/10], dtype=np.float32)

    #return np.array([pose.position.z_val/100],
    #                dtype=np.float32)

    #return np.array([abs(pose.position.z_val)/100], dtype=np.float32)
    
    #return np.array([self.client.getImuData().angular_velocity.x_val/10, self.client.getImuData().angular_velocity.y_val/10, self.client.getImuData().angular_velocity.z_val/10,
    #                  self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
    #                  self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6], dtype=np.float32)

  '''Returns the reward, given the pose (state)
  '''
  def reward_function(self, pose):
    #loss = abs(pose.position.z_val-self.initial_pose.position.z_val)/10 + abs(pose.position.x_val-self.initial_pose.position.x_val)/10 + abs(pose.position.y_val-self.initial_pose.position.y_val)/10 # Manhattan distance from initial pose
    #scaled_loss = np.log(1.5+loss) # apply logarithmic scaling to have values around 0.4-2.6 not changing too much
    #reward = -loss
    reward = 0
    if abs(pose.position.x_val-self.initial_pose.position.x_val) < 0.5: reward += 2
    elif abs(pose.position.x_val-self.initial_pose.position.x_val) < 1: reward += 1
    if abs(pose.position.y_val-self.initial_pose.position.y_val) < 0.5: reward += 2
    elif abs(pose.position.y_val-self.initial_pose.position.y_val) < 1: reward += 1
    if abs(pose.position.z_val-self.initial_pose.position.z_val) < 0.5: reward += 2
    elif abs(pose.position.z_val-self.initial_pose.position.z_val) < 1: reward += 1

    #if(abs(pose.position.x_val-self.initial_pose.position.x_val) < 0.1 ): reward = reward + 100
    #if(abs(pose.position.y_val-self.initial_pose.position.y_val) < 0.1 ): reward = reward + 100
    #if(abs(pose.position.z_val-self.initial_pose.position.z_val) < 0.1 ): reward = reward + 100
    #if(abs(pose.position.x_val-self.initial_pose.position.x_val) < 0.5 ): reward = reward + 20
    #if(abs(pose.position.y_val-self.initial_pose.position.y_val) < 0.5 ): reward = reward + 20
    #if(abs(pose.position.z_val-self.initial_pose.position.z_val) < 0.5 ): reward = reward + 20
    #if(abs(pose.position.x_val-self.initial_pose.position.x_val) < 1 ): reward = reward + 10
    #if(abs(pose.position.y_val-self.initial_pose.position.y_val) < 1 ): reward = reward + 10
    #if(abs(pose.position.z_val-self.initial_pose.position.z_val) < 1 ): reward = reward + 10
    #if(abs(pose.position.x_val-self.initial_pose.position.x_val) < 2 ): reward = reward + 5
    #if(abs(pose.position.y_val-self.initial_pose.position.y_val) < 2 ): reward = reward + 5
    #if(abs(pose.position.z_val-self.initial_pose.position.z_val) < 2 ): reward = reward + 5
    #if(abs(pose.position.x_val-self.initial_pose.position.x_val) < 3 ): reward = reward + 1
    #if(abs(pose.position.y_val-self.initial_pose.position.y_val) < 3 ): reward = reward + 1
    #if(abs(pose.position.z_val-self.initial_pose.position.z_val) < 3 ): reward = reward + 1

    return reward