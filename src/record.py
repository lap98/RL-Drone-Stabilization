#################################################
# Imports
#################################################

import os
import time
import datetime
import numpy as np
import tensorflow as tf
from lib.plotters import Plotter
from lib.customEnvironment_v0_8 import DroneEnvironment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import TimeLimit
import airsim   
import PySimpleGUI as sg      

np.random.seed(1234)
tf.random.set_seed(12345)

#################################################
# Reinforcement Learning parameters
#################################################

#save_path = 'C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/training_data/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Evaluation
eval_env_steps_limit = 400 # maximum number of steps in the TimeLimit of the evaluation environment
num_eval_episodes = 1


#################################################
# Environments instantiation
#################################################

eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(False, False), duration=eval_env_steps_limit)) # set limit to m steps in the environment


#################################################
# Agent
#################################################

global_step = tf.compat.v1.train.get_or_create_global_step() # global counter of the steps

policies = os.getcwd() + '/training_data/data/policies'


# data_plotter = Plotter()



def evaluate_agent(window, policy, eval_tf_env, num_eval_episodes):
  total_reward = 0
  for idx in range(num_eval_episodes):
    start = time.time()
    time_step = eval_tf_env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      # airsim.MultirotorClient().simSetWind(get_wind_vector(window))
      time_step = eval_tf_env.step(action_step.action)
      total_reward += float(time_step.reward)
    end = time.time()
  return total_reward / num_eval_episodes # avg reward per episode


def display_info(window):
  event, values = window.read(timeout=0)
  if event == sg.WIN_CLOSED:
      window.close()
  window['-EPOCH-'].update(epoch)
  window['-REWARD-'].update(avg_rew)


#################################################
# Window
#################################################
# Define the window's contents

layout = [[sg.Text("Epoch: ", key='-EPOCH-')],
          [sg.Text("Reward: ", key='-REWARD-')]
          ]

# Create the window
window = sg.Window('Window Title', layout)
avg_rewards = np.empty((0,2))


#################################################
# Load policies
#################################################
epoch = 0
avg_rew = 0
for policy in os.listdir(policies):
  print(policy)
  #display_info(window)
  saved_policy = tf.compat.v2.saved_model.load(policies + '/' + policy)
  avg_rew = evaluate_agent(window,saved_policy, eval_tf_env, num_eval_episodes)
  epoch = epoch + 50




#################################################
# TEvaluation
#################################################
#avg_rew = evaluate_agent(window,saved_policy, eval_tf_env, num_eval_episodes)
#avg_rewards = np.concatenate((avg_rewards, [[epoch, avg_rew]]), axis=0)
#data_plotter.update_eval_reward(avg_rew, eval_interval)
#np.save(save_path+'/avg_rewards.npy', avg_rewards)
#data_plotter.plot_evaluation_rewards(avg_rewards, save_path)

# Finish up by removing from the screen
window.close()         