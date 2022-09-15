#################################################
# Imports
#################################################

import os
import time
import airsim
import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


#################################################
# AirSim environment definition
#################################################

class DroneEnvironment(py_environment.PyEnvironment):

  '''Initializes the environment, connecting to the drone and setting the observation and action spaces
  :param enable_wind: whether to enable the wind
  :param randomize_initial_pose: whether to randomize the initial position and orientation of the drone
  :param save_path: the path of the data saving folder:
                      If present, the software saves all the states of the drone, for later analysis
                      If None, nothing is saved
  '''
  def __init__(self, enable_wind=False, randomize_initial_pose=False, save_path=None):
    self.enable_wind = enable_wind
    self.randomize_initial_pose = randomize_initial_pose
    self.save_path = save_path
    self._states_arr = None

    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    
    self._observation_spec = array_spec.ArraySpec(shape=(19,),dtype=np.float32, name='observation') # Observation (drone state) space
    self._action_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, name='action', minimum=0.0, maximum=1.0) # Action space: control motors power
    
    #self._state, _, _, _, _, _, _ = self.getState()
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
    or_stddev = 0.3 # in [rad], 0.15 -> at most around 20 deg of inclination
    if random:
      if random_uniform:
        u = np.random.uniform()
        v = np.random.uniform()
        w = np.random.uniform()
        new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0 + np.random.normal(0, pos_stddev), 0.0 + np.random.normal(0, pos_stddev), -100.0 + np.random.normal(0, pos_stddev)),
                                orientation_val=airsim.Quaternionr(np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v), np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)))
      else:
        new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0 + np.random.normal(0, pos_stddev), 0.0 + np.random.normal(0, pos_stddev), -100.0 + np.random.normal(0, pos_stddev)),
                                #orientation_val=airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)) # do not put noise in orientation
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
    self.client.takeoffAsync(timeout_sec=0.1, vehicle_name="SimpleFlight").join() # needed to enable the multirotor correct flying state

    new_pose, self.initial_pose = self.getNewPose(self.randomize_initial_pose)

    self.client.simSetVehiclePose(pose=new_pose, ignore_collision=False, vehicle_name="SimpleFlight")
    if self.client.getMultirotorState().landed_state == airsim.LandedState.Landed: print("[LANDED: Physics Engine NOT Engaged]")
    else: print("[CORRECTLY FLYING: Physics Engine Engaged]")
    time.sleep(0.01) # needed because otherwise, the neural net is too fast and the _steps begin too soon, when the drone is not ready, and no experience would be gained (rew=0)
  
  '''Sets a random wind in the simulation, with the given standard deviation in [m]
  '''
  def setRandomWind(self, stddev=2.5):
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
    self._steps = 0
    if self.save_path is not None: # saves states of the drone for later analysis
      if self._states_arr is not None:
        if not os.path.exists(self.save_path+'/states'): os.makedirs(self.save_path+'/states')
        np.save(self.save_path+'/states/'+str(time.time()), self._states_arr)
      self._states_arr = np.empty((0,19))

    if self.enable_wind: self.setRandomWind()
    self.reset_pose()

    self._state, _, _, _, _, _, _ = self.getState()
    self._episode_ended = False
    return ts.restart(self._state)



  '''Moves the drone as specified by the action, first checking for the termination conditions
  :param action: the tensorflow action, as described by the array spec in the __init__ function
  :param duration: how long the duration of the action has to be. If the movemetnt is continuous, only specifies the maximum duration (it is asyncronous)
  :param continuous: whether to do continuous movements or wait for the action to end before going back to the network inference
  :return: True if the episode has to end due to collisions or other, False otherwise
  '''
  def move(self, action, duration=1.002, continuous=True):
    #if self.client.simGetCollisionInfo().has_collided or self.client.simGetVehiclePose().position.z_val > -10: return True

    # With the join, the rotor thrust gets reset to the default hovering stable 2.42 for the time afterwards. Without, it is left to what the network has decided
    if continuous == True: # continuous movements -> the control loop is: network inference -> perform action asynchronously -> ...
      scale = 2.5 # scale for the delta thrust of each motor, how much importance to give it
      b_th = 0.59
      d_th = scale * (action-0.5) / 5 # delta thrust on the rotors
      th = np.clip(b_th + d_th, 0, 1) # thrust bias + delta, clipped
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(th[0]), rear_left_pwm=float(th[1]), front_left_pwm=float(th[2]), rear_right_pwm=float(th[3]), duration=duration)
      time.sleep(0.002)

    else: # discrete movements -> the control loop is: network inference -> perform action and join -> ...
      #self.client.hoverAsync()
      scale = 2.0 # scale for the delta thrust of each motor, how much importance to give it
      b_th = 0.59
      d_th = scale * (action-0.5) / 5 # delta thrust on the rotors
      th = np.clip(b_th + d_th, 0, 1) # thrust bias + delta, clipped
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(th[0]), rear_left_pwm=float(th[1]), front_left_pwm=float(th[2]), rear_right_pwm=float(th[3]), duration=duration).join()
    
    return False

  '''Override, one step: performs an action, retrieves the reward, returns either the transition or the termination signals associated
  :param action: the action to perform decided by the agent
  :return: either a transition or a termination
  '''
  def _step(self, action):
    # If the episode is done, reset the environment
    if self._episode_ended: return self.reset()

    # Perform the chosen move (if collision occurs, it returns True), get new state, compute reward
    end_now = self.move(action=action)
    self._state, pos, orient, ang_acc, ang_vel, lin_acc, lin_vel = self.getState()
    if self.save_path is not None: self._states_arr = np.concatenate((self._states_arr, [self._state]), axis=0) # save the states of the drone for later analysis
    reward = self.reward_function(pos, orient, ang_acc, ang_vel, lin_acc, lin_vel)

    # Handle states - step_type: 0->beginning, 1->normal transition, 2->terminal
    if end_now:
      print('Collision occurred or episode termination condition met')
      self._episode_ended = True
      reward = 0 # a collision occured, give a NEGATIVE? reward
      return ts.termination(self._state, reward=reward) # return terminal state to the agent
    else: # if were still going on we transition to the next state
      self._total_reward += reward
      return ts.transition(self._state, reward=reward)



  '''Returns the state as a numpy array with float32 values
  '''
  def getState(self):
    state   = self.client.getMultirotorState()
    pos     = state.kinematics_estimated.position
    orient  = state.kinematics_estimated.orientation
    ang_acc = state.kinematics_estimated.angular_acceleration
    ang_vel = state.kinematics_estimated.angular_velocity
    lin_acc = state.kinematics_estimated.linear_acceleration
    lin_vel = state.kinematics_estimated.linear_velocity
    
    return np.array([(pos.z_val-self.initial_pose.position.z_val)/10, (pos.x_val-self.initial_pose.position.x_val)/10, (pos.y_val-self.initial_pose.position.y_val)/10, # between -1 and 1 more or less
                    orient.w_val, orient.x_val, orient.y_val, orient.z_val, # between -1 and 1
                    ang_acc.x_val/10, ang_acc.y_val/10, ang_acc.z_val/10,   # at most around 30ish?
                    ang_vel.x_val/10, ang_vel.y_val/10, ang_vel.z_val/10,   # at most around 30ish?
                    lin_acc.x_val/10, lin_acc.y_val/10, lin_acc.z_val/10,   # at most around 30ish?
                    lin_vel.x_val/10, lin_vel.y_val/10, lin_vel.z_val/10],  # at most around 30ish?
                    dtype=np.float32), pos, orient, ang_acc, ang_vel, lin_acc, lin_vel
  
  '''Returns the reward, given the pose (state)
  '''
  def reward_function(self, pos, orient, ang_acc, ang_vel, lin_acc, lin_vel):

    reward = max(0, 1 - np.sqrt((pos.z_val-self.initial_pose.position.z_val)**2 + (pos.x_val-self.initial_pose.position.x_val)**2 + (pos.y_val-self.initial_pose.position.y_val)**2))
    reward -= 0.1 * np.sqrt((orient.w_val-1)**2 + orient.x_val**2 + orient.y_val**2 + orient.z_val**2)
    reward -= 0.1 * np.sqrt(ang_vel.x_val**2 + ang_vel.y_val**2 + ang_vel.z_val**2)

    self._steps += 1
    #if self._steps % 10 == 0: print('Position of drone: <', pos.x_val, pos.y_val, pos.z_val, '>')

    return reward