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
num_eval_episodes = 10


#################################################
# Environments instantiation
#################################################

eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(False, False), duration=eval_env_steps_limit)) # set limit to m steps in the environment


#################################################
# Agent
#################################################

global_step = tf.compat.v1.train.get_or_create_global_step() # global counter of the steps

#################################################
# Load policy
#################################################

policy_path = os.getcwd() +'/working/working_td3_2/20220911_185243/policies'
saved_policy = tf.compat.v2.saved_model.load(policy_path)

#################################################
# Training and Evaluation functions
#################################################

#data_plotter = Plotter()

def get_wind_vector(window):
  event, values = window.read(timeout=0)
  print(event, values)
  if event == sg.WIN_CLOSED or event == 'Exit':
    window.close()
  window['-XLEFT-'].update(int(values['-XSLIDER-']))
  window['-XRIGHT-'].update(int(values['-XSLIDER-']))
  window['-YLEFT-'].update(int(values['-YSLIDER-']))
  window['-YRIGHT-'].update(int(values['-YSLIDER-']))
  window['-ZLEFT-'].update(int(values['-ZSLIDER-']))
  window['-ZRIGHT-'].update(int(values['-ZSLIDER-']))
  if event == 'Show':
      sg.popup(f'The slider value = {values["-SLIDER-"]}')
  return airsim.Vector3r(values["-XSLIDER-"],values["-YSLIDER-"],values["-ZSLIDER-"])



def evaluate_agent(window, policy, eval_tf_env, num_eval_episodes):
  print('\nEVALUATING *******\n')
  total_reward = 0
  for idx in range(num_eval_episodes):
    print('Evaluation iteration:', idx)
    start = time.time()
    time_step = eval_tf_env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      airsim.MultirotorClient().simSetWind(get_wind_vector(window))
      time_step = eval_tf_env.step(action_step.action)
      total_reward += float(time_step.reward)
    end = time.time()
    print('Control loop timing for 1 timestep [s]:', (end-start)/eval_env_steps_limit)
  print('\n******* EVALUATION ENDED\n')
  return total_reward / num_eval_episodes # avg reward per episode

#################################################
# Window
#################################################
# Define the window's contents

layout = [[sg.Text('Wind component along the X axis')],
          [sg.T('0',size=(4,1), key='-XLEFT-'),
           sg.Slider((-50,50),default_value=0, key='-XSLIDER-', orientation='h', enable_events=True, disable_number_display=True),
           sg.T('0', size=(4,1), key='-XRIGHT-')],
          [sg.Text('Wind component along the Y axis')],
          [sg.T('0',size=(4,1), key='-YLEFT-'),
           sg.Slider((-50,50),default_value=0, key='-YSLIDER-', orientation='h', enable_events=True, disable_number_display=True),
           sg.T('0', size=(4,1), key='-YRIGHT-')],
          [sg.Text('Wind component along the Z axis')],
          [sg.T('0',size=(4,1), key='-ZLEFT-'),
           sg.Slider((-50,50),default_value=0, key='-ZSLIDER-', orientation='h', enable_events=True, disable_number_display=True),
           sg.T('0', size=(4,1), key='-ZRIGHT-')],
          [ sg.Button('Exit')]]

# Create the window
window = sg.Window('Window Title', layout)
avg_rewards = np.empty((0,2))


#################################################
# TEvaluation
#################################################
avg_rew = evaluate_agent(window,saved_policy, eval_tf_env, num_eval_episodes)
#avg_rewards = np.concatenate((avg_rewards, [[epoch, avg_rew]]), axis=0)
#data_plotter.update_eval_reward(avg_rew, eval_interval)
#np.save(save_path+'/avg_rewards.npy', avg_rewards)
#data_plotter.plot_evaluation_rewards(avg_rewards, save_path)

# Finish up by removing from the screen
window.close()         