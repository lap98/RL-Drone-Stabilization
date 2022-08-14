# AirSim docs: https://microsoft.github.io/AirSim/api_docs/html/
# TF-agents library at the following link https://www.tensorflow.org/agents and the tutorial https://www.tensorflow.org/agents/tutorials/0_intro_rl
# How to setup the environment https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059
# py_environment https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment?hl=en#get_state
# https://towardsdatascience.com/cartpole-problem-using-tf-agents-build-your-first-reinforcement-learning-application-3e6006adeba7

#################################################
# Imports
#################################################
import airsim
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.agents import DdpgAgent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.metrics import tf_metrics



#################################################
# AirSim environment definition
#################################################
class DroneEnvironment(py_environment.PyEnvironment):

  def __init__(self):

    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)

    self.initialPose = self.client.simGetObjectPose(object_name="SimpleFlight")

    # Observation space: IMU (3x angular velocity, 3x linear acceleration), ultrasound distance from the ground, barometer;
    # Action space: control motors power
    self._action_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, name='action', minimum=0.0, maximum=1.0)
    self._observation_spec = array_spec.ArraySpec(shape=(8,),dtype=np.float32, name='observation')
  
    # The state of the environment which can be seen by the drone using its sensors; it represents also the input of the network
    self._state = np.array([self.client.getImuData().angular_velocity.x_val, self.client.getImuData().angular_velocity.y_val, self.client.getImuData().angular_velocity.z_val, 
                            self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
                            self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6])
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec
  
  def observation_spec(self):
    return self._observation_spec

  # Reset the custom environment we created
  def _reset(self):
    self._episode_ended = False
    self.client.reset()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    pose = self.client.simGetVehiclePose()
    pose.position.z_val = -5.0
    self.client.simSetVehiclePose(pose, ignore_collision=False)
    #self.client.takeoffAsync().join()
    self.initialPose = self.client.simGetObjectPose(object_name="SimpleFlight")
    self._state = [self.client.getImuData().angular_velocity.x_val, self.client.getImuData().angular_velocity.y_val, self.client.getImuData().angular_velocity.z_val,
                  self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
                  self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6]
    return ts.restart(np.array(self._state, dtype=np.float32))

  def _step(self, action):
    # If the episode is done, reset the environment
    if self._episode_ended: return self.reset()

    # Perform the chosen move (if collision occurs, it returns True), get new state, compute reward
    collision = self.move(action=action)
    self._state = self.getState()
    reward = self.reward_function(self.client.simGetObjectPose(object_name="SimpleFlight"))

    # Handle states - step_type: 0->beginning, 1->normal transition, 2->terminal
    if collision:
      self._episode_ended = True
      reward = -1 # a collision occured, give a negative reward
      return ts.termination(np.array(self._state, dtype=np.float32), reward) # return terminal state to the agent
    else: # if were still going on we transition to the next state
      return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=0.9)

  def move(self, action, duration=0.1, continuous=False):
    if self.client.simGetCollisionInfo().has_collided: return True

    if continuous == False: # discrete actions
      self.client.hoverAsync()
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(action[0]), rear_left_pwm=float(action[1]), front_left_pwm=float(action[2]), rear_right_pwm=float(action[3]), duration=duration)
      if self.client.simGetCollisionInfo().has_collided: return True
      return False
    else: # continuous movements
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(action[0]), rear_left_pwm=float(action[1]), front_left_pwm=float(action[2]), rear_right_pwm=float(action[3]), duration=20)
      if self.client.simGetCollisionInfo().has_collided: return True
      return False

  def getState(self):
    return np.array([self.client.getImuData().angular_velocity.x_val, self.client.getImuData().angular_velocity.y_val, self.client.getImuData().angular_velocity.z_val,
                      self.client.getImuData().linear_acceleration.x_val, self.client.getImuData().linear_acceleration.y_val, self.client.getImuData().linear_acceleration.z_val,
                      self.client.getBarometerData(vehicle_name="SimpleFlight").pressure/101325,self.client.getDistanceSensorData(vehicle_name="SimpleFlight").distance/6])

  def reward_function(self,pose):
    reward = 1/(1+np.sqrt((pose.position.x_val - self.initialPose.position.x_val)**2+(pose.position.y_val - self.initialPose.position.y_val)**2+(pose.position.z_val - self.initialPose.position.z_val)**2))
    reward = reward + 1/(1+np.sqrt((pose.orientation.w_val - self.initialPose.orientation.w_val)**2+(pose.orientation.x_val - self.initialPose.orientation.x_val)**2+(pose.orientation.y_val - self.initialPose.orientation.y_val)**2+(pose.orientation.z_val - self.initialPose.orientation.z_val)**2))
    return reward




#################################################
# Reinforcement Learning parameters
#################################################
# https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en

collect_steps_per_iteration = 300 # maximum number of steps in each episode
replay_buffer_capacity = 10000

fc_layer_params = (75, 40,)

epochs = 100
batch_size = 32
learning_rate = 1e-3
#log_interval = 5

#num_eval_episodes = 10
#eval_interval = 1000



#################################################
# Environment instantiation & Random Policy
#################################################

tf_env = tf_py_environment.TFPyEnvironment(DroneEnvironment())
# Environment testing code
#environment = DroneEnvironment()
#action = np.array([0.5,0.3,0.1,0.7], dtype=np.float32)
#time_step = environment.reset()
#print(time_step)
#while not time_step.is_last():
#  time_step = environment.step(action)
#  print(time_step)

tf_policy = random_tf_policy.RandomTFPolicy(action_spec=tf_env.action_spec(), time_step_spec=tf_env.time_step_spec())
# Policy testing code
#observation = tf.ones(environment.time_step_spec().observation.shape)
#time_step = ts.restart(observation)
#action_step = tf_policy.action(time_step)
#print('Action:',action_step.action)



#################################################
# Agent definition
#################################################

# Network https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/q_network/QNetwork
actor_net = ActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=fc_layer_params)
critic_net = CriticNetwork((tf_env.observation_spec(), tf_env.action_spec()), joint_fc_layer_params=fc_layer_params)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
agent = DdpgAgent(tf_env.time_step_spec(),
                  tf_env.action_spec(),
                  actor_network=actor_net,
                  critic_network=critic_net,
                  actor_optimizer=optimizer,
                  critic_optimizer=optimizer,
                  td_errors_loss_fn=common.element_wise_squared_loss)
agent.initialize()



#################################################
# Replay buffer & Collect Driver
#################################################

# Create the replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=replay_buffer_capacity)

# Create the collect driver
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [replay_buffer.add_batch, num_episodes, env_steps]
collect_driver = dynamic_step_driver.DynamicStepDriver(tf_env, agent.collect_policy, observers=observers, num_steps=collect_steps_per_iteration) # instead of tf_policy use the agent policy, which is the OUNoisePolicy
# Driver testing code; initial driver.run will reset the environment and initialize the policy.
#final_time_step, policy_state = collect_driver.run()
#print('final_time_step', final_time_step, 'Number of Steps: ', env_steps.result().numpy(), 'Number of Episodes: ', num_episodes.result().numpy())

# Initial data collection
print('Collecting initial data')
collect_driver.run()
print('Data collection executed')

# Transform Replay Buffer to Dataset
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3) # read batches of 32 elements, each with 2 timesteps
iterator = iter(dataset)



#################################################
# Training functions
#################################################

def train_one_iteration():
  collect_driver.run() # collect a few steps using collect_policy and save to the replay buffer
  experience, unused_info = next(iterator) # sample a batch of data from the buffer and update the agent's network
  with tf.device('/GPU:0'): train_loss = agent.train(experience)
  iteration = agent.train_step_counter.numpy()
  print ('iteration: {0} loss: {1}'.format(iteration, train_loss.loss))

# Training Loop
for epoch in range(epochs):
  train_one_iteration()