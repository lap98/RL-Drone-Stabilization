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
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.agents import PPOAgent
from tf_agents.policies import policy_saver
from tf_agents.utils import common

np.random.seed(1234)
tf.random.set_seed(12345)


#################################################
# Reinforcement Learning parameters
#################################################

save_path = 'C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/training_data/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# PPO is an on-policy algorithm, so maybe lower the initial data collection?
# Data collection
replay_buffer_capacity = 1000000
initial_collect_steps = 1000 # total number of steps collected with a random policy. Every time the steps TimeLimit is reached, the environment is reset

# Agent
fc_layer_params = (64, 64,)

# Training
train_env_steps_limit = 200 # maximum number of steps in the TimeLimit of the training environment
collect_steps_per_iteration = 200 # maximum number of steps in each episode

epochs = 4000
batch_size = 128
learning_rate = 1e-3
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

tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(), duration=train_env_steps_limit)) # set limit to 100 steps in the environment
eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(), duration=eval_env_steps_limit)) # 1000 steps duration


#################################################
# Agent
#################################################

global_step = tf.compat.v1.train.get_or_create_global_step()

actor_net = ActorDistributionNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.tanh)
value_net = ValueNetwork(tf_env.observation_spec(), fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.tanh)

agent = PPOAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 actor_net=actor_net,
                 value_net=value_net,
                 optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 discount_factor=0.99,
                 num_epochs=1,
                 train_step_counter=global_step)
agent.initialize()

print("Actor network summary and details")
print(actor_net.summary())
for i, layer in enumerate (actor_net.layers):
    print (i, layer)
    try: print ("    ",layer.activation)
    except AttributeError: print('   no activation attribute')

print("Critic network summary and details")
print(value_net.summary())
for i, layer in enumerate (value_net.layers):
    print (i, layer)
    try: print ("    ",layer.activation)
    except AttributeError: print('   no activation attribute')


#################################################
# Replay Buffer & Collect Driver
#################################################

# PPO is an on-policy algorithm, so it makes sense to collect only on-policy data

# Create the replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=replay_buffer_capacity)

# Create the collect driver
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [replay_buffer.add_batch, num_episodes, env_steps]
collect_driver = dynamic_step_driver.DynamicStepDriver(tf_env, agent.collect_policy, observers=observers, num_steps=initial_collect_steps) # use tf_policy, which is random

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
    tf_policy_saver.save(policy_dir) # policy saving for later restore
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