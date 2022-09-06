# AirSim docs: https://microsoft.github.io/AirSim/api_docs/html/
# TF-agents library at the following link https://www.tensorflow.org/agents and the tutorial https://www.tensorflow.org/agents/tutorials/0_intro_rl
# How to setup the environment https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059
# py_environment https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment?hl=en#get_state
# https://towardsdatascience.com/cartpole-problem-using-tf-agents-build-your-first-reinforcement-learning-application-3e6006adeba7

#################################################
# Imports
#################################################

import os
import airsim
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.agents import DdpgAgent
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.metrics import tf_metrics
from tf_agents.environments import TimeLimit

np.random.seed(1234)
tf.random.set_seed(12345)

#################################################
# AirSim environment definition
#################################################

class DroneEnvironment(py_environment.PyEnvironment):

  '''Initializes the environment, connecting to the drone and setting the observation and action spaces
  '''
  def __init__(self):
    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    # Observation space: IMU (3x angular velocity, 3x linear acceleration), ultrasound distance from the ground, barometer
    self._observation_spec = array_spec.ArraySpec(shape=(1,),dtype=np.float32, name='observation')
    # Action space: control motors power
    self._action_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, name='action', minimum=0.0, maximum=1.0)
    # The state of the environment which can be seen by the drone using its sensors; it represents also the input of the network
    self._state = self.getState()
    self._episode_ended = False
    self._total_reward = 0
  
  def action_spec(self):
    return self._action_spec
  
  def observation_spec(self):
    return self._observation_spec
  
  '''Resets the custom environment created
  '''
  def _reset(self):
    print('Total reward for the episode before:', self._total_reward)
    self._total_reward = 0

    self.client.reset()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)

    pose = self.client.simGetVehiclePose()
    pose.position.z_val = -100.0
    self.client.simSetVehiclePose(pose, ignore_collision=False)
    #self.client.takeoffAsync().join()
    self.initialPose = self.client.simGetObjectPose(object_name="SimpleFlight")

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
    reward = self.reward_function(self.client.simGetObjectPose(object_name="SimpleFlight"))

    # Handle states - step_type: 0->beginning, 1->normal transition, 2->terminal
    if end_now:
      print('Collision occurred or termination episode end met')
      self._episode_ended = True
      reward = 0 # a collision occured, give a NEGATIVE? reward
      return ts.termination(self._state, reward) # return terminal state to the agent
    else: # if were still going on we transition to the next state
      self._total_reward += reward
      return ts.transition(self._state, reward=reward)

  '''Moves the drone as specified by the action
  :param action: the tensorflow action, as described by the array spec in the __init__ function
  :param duration: how long the duration of the action has to be. If the movemetnt is continuous, only specifies the maximum duration (it is asyncronous)
  :param continuous: whether to do continuous movements or wait for the action to end before going back to the network inference
  :return: True if the episode has to end due to collisions or other, False otherwise
  '''
  def move(self, action, duration=0.1, continuous=True):
    if self.client.simGetCollisionInfo().has_collided or self.client.simGetVehiclePose().position.z_val > -10: return True

    if continuous == True: # continuous movements -> the control loop is: network inference -> perform action asynchronously -> ...
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(action[0]), rear_left_pwm=float(action[1]), front_left_pwm=float(action[2]), rear_right_pwm=float(action[3]), duration=duration)
    else: # discrete movements -> the control loop is: network inference -> perform action and join -> ...
      #self.client.hoverAsync()
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(action[0]), rear_left_pwm=float(action[1]), front_left_pwm=float(action[2]), rear_right_pwm=float(action[3]), duration=duration).join()
    
    return False

  '''Returns the state as a numpy array with float32 values
  '''
  def getState(self):
    pose = self.client.simGetVehiclePose()
    #return np.array([pose.position.x_val/10,pose.position.y_val/10,pose.position.z_val/10, pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val,
    #                  self.client.getImuData().angular_velocity.x_val/10, self.client.getImuData().angular_velocity.y_val/10, self.client.getImuData().angular_velocity.z_val/10,
    #                  self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val], dtype=np.float32)
    #return np.array([self.client.getImuData().angular_velocity.x_val/10, self.client.getImuData().angular_velocity.y_val/10, self.client.getImuData().angular_velocity.z_val/10], dtype=np.float32)
    return np.array([pose.position.z_val/10], dtype=np.float32)
    #return np.array([self.client.getImuData().angular_velocity.x_val/10, self.client.getImuData().angular_velocity.y_val/10, self.client.getImuData().angular_velocity.z_val/10,
    #                  self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
    #                  self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6], dtype=np.float32)

  '''Returns the reward, given the pose (state)
  '''
  def reward_function(self, pose):
    #reward = 1/(0.0001+np.sqrt((pose.position.x_val - self.initialPose.position.x_val)**2+(pose.position.y_val - self.initialPose.position.y_val)**2+(pose.position.z_val - self.initialPose.position.z_val)**2))
    #reward = reward + 1/(0.0001+np.sqrt((pose.orientation.w_val - self.initialPose.orientation.w_val)**2+(pose.orientation.x_val - self.initialPose.orientation.x_val)**2+(pose.orientation.y_val - self.initialPose.orientation.y_val)**2+(pose.orientation.z_val - self.initialPose.orientation.z_val)**2))
    #reward = - (abs(self.client.getImuData().angular_velocity.x_val)/10 + abs(self.client.getImuData().angular_velocity.y_val)/10 + abs(self.client.getImuData().angular_velocity.z_val)/10)
    reward = abs(pose.position.z_val)/100

    return reward




#################################################
# Reinforcement Learning parameters
#################################################

# https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en

# Data collection
replay_buffer_capacity = 100000
initial_collect_steps = 1000 # total number of steps collected with a random policy. Every time the steps TimeLimit is reached, the environment is reset

# Agent
fc_layer_params = (64, 64,)

# Training
train_env_steps_limit = 100 # maximum number of steps in the TimeLimit of the training environment
collect_steps_per_iteration = 100 # maximum number of steps in each episode

epochs = 150
batch_size = 128
learning_rate = 1e-3
checkpoint_dir = 'C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/ckpts/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
policy_dir = 'C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/policies/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ckpts_interval = 10 # every how many epochs to store a checkpoint during training

# Evaluation
eval_env_steps_limit = 1000 # maximum number of steps in the TimeLimit of the evaluation environment
num_eval_episodes = 5
eval_interval = 50 # interval for evaluation and policy saving, =epochs for evaluation only at the end


#################################################
# Environments instantiation
#################################################

tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(), duration=train_env_steps_limit)) # set limit to 100 steps in the environment
eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(), duration=eval_env_steps_limit)) # 1000 steps duration
# Environment testing code
#environment = DroneEnvironment()
#action = np.array([0.5,0.3,0.1,0.7], dtype=np.float32)
#time_step = environment.reset()
#print(time_step)
#while not time_step.is_last():
#  time_step = environment.step(action)
#  print(time_step)


#################################################
# Agent
#################################################

global_step = tf.compat.v1.train.get_or_create_global_step()

# Network https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/q_network/QNetwork
actor_net = ActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.sigmoid)
critic_net = CriticNetwork((tf_env.observation_spec(), tf_env.action_spec()), joint_fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.sigmoid)
agent = DdpgAgent(tf_env.time_step_spec(),
                  tf_env.action_spec(),
                  actor_network=actor_net,
                  critic_network=critic_net,
                  actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                  critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                  #td_errors_loss_fn=common.element_wise_squared_loss,
                  gamma=0.99,
                  train_step_counter=global_step)
agent.initialize()


#################################################
# Random Policy
#################################################

tf_policy = random_tf_policy.RandomTFPolicy(action_spec=tf_env.action_spec(), time_step_spec=tf_env.time_step_spec())
# Policy testing code
#observation = tf.ones(environment.time_step_spec().observation.shape)
#time_step = ts.restart(observation)
#action_step = tf_policy.action(time_step)
#print('Action:',action_step.action)


#################################################
# Replay Buffer & Collect Driver
#################################################

# Create the replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=replay_buffer_capacity)

# Create the collect driver
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [replay_buffer.add_batch, num_episodes, env_steps]
collect_driver = dynamic_step_driver.DynamicStepDriver(tf_env, tf_policy, observers=observers, num_steps=initial_collect_steps) # use tf_policy, which is random
# Driver testing code; initial driver.run will reset the environment and initialize the policy.
#final_time_step, policy_state = collect_driver.run()
#print('final_time_step', final_time_step, 'Number of Steps: ', env_steps.result().numpy(), 'Number of Episodes: ', num_episodes.result().numpy())

# Initial data collection
print('Collecting initial data')
collect_driver.run()
print('Data collection executed')

train_driver = dynamic_step_driver.DynamicStepDriver(tf_env, agent.collect_policy, observers=observers, num_steps=collect_steps_per_iteration) # instead of tf_policy use the agent.collect_policy, which is the OUNoisePolicy

# Transform Replay Buffer to Dataset
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3) # read batches of 32 elements, each with 2 timesteps
iterator = iter(dataset)


#################################################
# Training and Evaluation functions
#################################################

train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, max_to_keep=1, agent=agent, policy=agent.policy, replay_buffer=replay_buffer, global_step=global_step)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

def train_one_iteration():
  start = datetime.datetime.now()
  train_driver.run() # collect a few steps using collect_policy and save to the replay buffer
  end = datetime.datetime.now()
  print('Control loop timing, for 1 step:', (end-start)/collect_steps_per_iteration)
  experience, unused_info = next(iterator) # sample a batch of data from the buffer and update the agent's network
  with tf.device('/GPU:0'): train_loss = agent.train(experience) # trains on 1 batch of experience
  iteration = agent.train_step_counter.numpy()
  print ('Iteration: {0}, loss: {1}'.format(iteration, train_loss.loss))

def evaluate_agent(policy, eval_tf_env, num_eval_episodes):
  print('\nEVALUATING *******\n')
  total_reward = 0
  for idx in range(num_eval_episodes):
    print('Evaluation iteration:', idx)
    time_step = eval_tf_env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_tf_env.step(action_step.action)
      total_reward += float(time_step.reward)
  print('\n******* EVALUATION ENDED\n')
  return total_reward / num_eval_episodes # avg reward per episode

# Training loop, evaluation & checkpoints saving
avg_rewards = np.empty((0,2))
for epoch in range(epochs+1):
  train_one_iteration()
  if epoch % ckpts_interval == 0:
    train_checkpointer.save(global_step)
  if epoch % eval_interval == 0:
    tf_policy_saver.save(policy_dir) # policy saving for later restore
    avg_rew = evaluate_agent(agent.policy, eval_tf_env, num_eval_episodes)
    avg_rewards = np.concatenate((avg_rewards, [[epoch, avg_rew]]), axis=0)

plt.figure()
plt.title('Policy evaluation rewards vs epochs')
plt.plot(avg_rewards[:,0], avg_rewards[:,1])
plt.xlabel('Training epoch')
plt.ylabel('Average evaluation reward')
plt.show()

# Restoring a checkpoint
#train_checkpointer.initialize_or_restore()
#global_step = tf.compat.v1.train.get_global_step()

# Restoring only the policy
#saved_policy = tf.saved_model.load(policy_dir)