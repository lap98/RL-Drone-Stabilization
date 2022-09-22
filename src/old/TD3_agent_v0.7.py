#################################################
# Imports
#################################################

import time
import datetime
import numpy as np
import tensorflow as tf
from lib.plotters import Plotter
from lib.customEnvironment_v0_8 import DroneEnvironment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import TimeLimit
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks import network
from tf_agents.networks import encoding_network
from tf_agents.networks import utils
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents import Td3Agent
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.utils import nest_utils

np.random.seed(1234)
tf.random.set_seed(12345)


#################################################
# Reinforcement Learning parameters
#################################################

save_path = 'C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/training_data/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Data collection
replay_buffer_capacity = 1000000
initial_collect_steps = 5000 # total number of steps collected with a random policy. Every time the steps TimeLimit is reached, the environment is reset

# Agent
fc_layer_params = (128, 128,)

# Training
train_env_steps_limit = 200 # maximum number of steps in the TimeLimit of the training environment
collect_steps_per_iteration = 200 # maximum number of steps in each episode

epochs = 3000
batch_size = 512
learning_rate = 3e-4
checkpoint_dir = save_path + '/ckpts'
policy_dir = save_path + '/policies'
ckpts_interval = 10 # every how many epochs to store a checkpoint during training

# Evaluation
eval_env_steps_limit = 400 # maximum number of steps in the TimeLimit of the evaluation environment
num_eval_episodes = 5
eval_interval = 50 # interval for evaluation and policy saving, =epochs for evaluation only at the end


#################################################
# Environments instantiation
#################################################

tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(False, True), duration=train_env_steps_limit)) # set limit to n steps in the environment
eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(False, True, save_path), duration=eval_env_steps_limit)) # set limit to m steps in the environment


#################################################
# Agent
#################################################

global_step = tf.compat.v1.train.get_or_create_global_step() # global counter of the steps

class ActorNetworkCustom(network.Network):
  '''Defines a custom network for the actor
  '''
  def __init__(self, observation_spec, action_spec, name='ActorNetworkCustom'):
    super(ActorNetworkCustom, self).__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)

    # Multiple float output
    self._action_spec = action_spec
    # Initialize the custom tf layers here:
    self._dense1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='Dense1')
    self._dense2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='Dense2')
    self._add1 = tf.keras.layers.Add()
    self._dense3 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='Dense3')
    self._add2 = tf.keras.layers.Add()
    self._dense4 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='Dense4')
    self._add3 = tf.keras.layers.Add()
    self._action_projection_layer = tf.keras.layers.Dense(self._action_spec.shape[0], activation=tf.keras.activations.sigmoid, name='action')

  def call(self, observations, step_type=(), network_state=()):
    # We use batch_squash here in case the observations have a time sequence component
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    # Forward pass through the custom tf layers here (defined above):
    x = self._dense1(observations)
    res1 = self._dense2(x)
    x = self._add1([x, res1])
    res2 = self._dense3(x)
    x = self._add2([x, res2])
    res3 = self._dense4(x)
    x = self._add3([x, res3])
    actions = self._action_projection_layer(x)

    # Scale action to expected spec and return the action
    actions = common.scale_to_spec(actions, self._action_spec)
    return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state

class CriticNetworkCustom(network.Network):
  '''Defines a custom network for the critic
  '''
  def __init__(self, observation_spec, action_spec, name='CriticNetworkCustom'):
    # Invoke constructor of network.Network
    super(CriticNetworkCustom, self).__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)

    # Given state and action, retrieve Q value
    self._obs_spec = observation_spec
    self._action_spec = action_spec
    # Encoding layer concatenates state and action inputs, adds dense layer:
    self._encoder = encoding_network.EncodingNetwork((observation_spec, action_spec),
                                                     fc_layer_params=(128,),
                                                     preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1),
                                                     activation_fn = tf.keras.activations.relu,
                                                     batch_squash=True)
    # Initialize the custom tf layers here:
    self._dense2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='Dense2')
    self._add1 = tf.keras.layers.Add()
    self._dense3 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='Dense3')
    self._add2 = tf.keras.layers.Add()
    self._dense4 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='Dense4')
    self._add3 = tf.keras.layers.Add()
    self._value_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear, name='Value') # Q-function output

  def call(self, observations, step_type=(), network_state=()):
    # Forward pass through the custom tf layers here (defined above):
    state, network_state = self._encoder(observations, step_type=step_type, network_state=network_state)
    res1 = self._dense2(state)
    x = self._add1([state, res1])
    res2 = self._dense3(x)
    x = self._add2([x, res2])
    res3 = self._dense4(x)
    x = self._add3([x, res3])
    value = self._value_layer(x)

    return tf.reshape(value,[-1]), network_state

actor_net = ActorNetworkCustom(tf_env.observation_spec(), tf_env.action_spec())#ActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.tanh)
critic_net = CriticNetworkCustom(tf_env.observation_spec(), tf_env.action_spec())#CriticNetwork((tf_env.observation_spec(), tf_env.action_spec()), joint_fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.relu)

agent = Td3Agent(tf_env.time_step_spec(),
                  tf_env.action_spec(),
                  actor_network=actor_net,
                  critic_network=critic_net,
                  actor_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  critic_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  target_update_tau=1.0,
                  target_update_period=2,
                  gamma=0.99,
                  train_step_counter=global_step)

agent.initialize()

print("\nActor network summary and details")
print(actor_net.summary())
for i, layer in enumerate (actor_net.layers):
    print (i, layer)
    try: print ("    ",layer.activation)
    except AttributeError: print('   no activation attribute')

print("\nCritic network summary and details")
print(critic_net.summary())
for i, layer in enumerate (critic_net.layers):
    print (i, layer)
    try: print ("    ",layer.activation)
    except AttributeError: print('   no activation attribute')


#################################################
# Replay Buffer & Collect Driver
#################################################

# Initial collect policy - random
tf_policy = random_tf_policy.RandomTFPolicy(action_spec=tf_env.action_spec(), time_step_spec=tf_env.time_step_spec())

# Create the replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=replay_buffer_capacity)

# Create the initial and training collect drivers
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [replay_buffer.add_batch, num_episodes, env_steps]
collect_driver = dynamic_step_driver.DynamicStepDriver(tf_env, tf_policy, observers=observers, num_steps=initial_collect_steps) # use tf_policy, which is random

train_driver = dynamic_step_driver.DynamicStepDriver(tf_env, agent.collect_policy, observers=observers, num_steps=collect_steps_per_iteration) # instead of tf_policy use the agent.collect_policy, which is the OUNoisePolicy

# Initial data collection
print('\nCollecting initial data')
collect_driver.run()
print('Data collection executed\n')

# Transform Replay Buffer to Dataset
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3) # read batches of 32 elements, each with 2 timesteps
iterator = iter(dataset)


#################################################
# Training and Evaluation functions
#################################################

train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, max_to_keep=1, agent=agent, policy=agent.policy, replay_buffer=replay_buffer, global_step=global_step)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

data_plotter = Plotter()

def train_one_iteration():
  start = time.time()
  train_driver.run() # collect a few steps using collect_policy and save to the replay buffer
  end = time.time()
  experience, unused_info = next(iterator) # sample a batch of data from the buffer and update the agent's network
  with tf.device('/CPU:0'): train_loss = agent.train(experience) # trains on 1 batch of experience
  iteration = agent.train_step_counter.numpy()
  #data_plotter.update_loss(train_loss.loss)
  print ('Iteration:', iteration)
  print('Total_loss:', float(train_loss.loss), 'actor_loss:', float(train_loss.extra.actor_loss), 'critic_loss:', float(train_loss.extra.critic_loss))
  print('Control loop timing for 1 timestep [s]:', (end-start)/collect_steps_per_iteration)

def evaluate_agent(policy, eval_tf_env, num_eval_episodes):
  print('\nEVALUATING *******\n')
  total_reward = 0
  for idx in range(num_eval_episodes):
    print('Evaluation iteration:', idx)
    start = time.time()
    time_step = eval_tf_env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_tf_env.step(action_step.action)
      total_reward += float(time_step.reward)
    end = time.time()
    print('Control loop timing for 1 timestep [s]:', (end-start)/eval_env_steps_limit)
  print('\n******* EVALUATION ENDED\n')
  return total_reward / num_eval_episodes # avg reward per episode

# Training loop, evaluation & checkpoints saving
avg_rewards = np.empty((0,2))
for epoch in range(epochs+1):
  train_one_iteration()
  if epoch % ckpts_interval == 0:
    train_checkpointer.save(global_step)
  if epoch % eval_interval == 0:
    tf_policy_saver.save(policy_dir+'/'+str(time.time())) # policy saving for later restore
    avg_rew = evaluate_agent(agent.policy, eval_tf_env, num_eval_episodes)
    avg_rewards = np.concatenate((avg_rewards, [[epoch, avg_rew]]), axis=0)
    data_plotter.update_eval_reward(avg_rew, eval_interval)

np.save(save_path+'/avg_rewards.npy', avg_rewards)

data_plotter.plot_evaluation_rewards(avg_rewards, save_path)

# Restoring a checkpoint
#train_checkpointer.initialize_or_restore()
#global_step = tf.compat.v1.train.get_global_step()

# Restoring only the policy
#saved_policy = tf.saved_model.load(policy_dir)