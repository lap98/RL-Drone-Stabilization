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
import pygad.gann
import pygad.nn

np.random.seed(1234)
tf.random.set_seed(12345)


#################################################
# Genetic Optimization parameters
#################################################

save_path = 'C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/training_data/' + '20220913_143534'

# Agent
fc_layer_params = [64,]
activations = ["relu"]

# Training
epochs = 100
checkpoint_dir = save_path + '/ckpts'
policy_dir = save_path + '/policies'
ckpts_interval = 10 # every how many epochs to store a checkpoint during training

# Evaluation
eval_env_steps_limit = 200 # maximum number of steps in the TimeLimit of the evaluation environment


#################################################
# Environments instantiation
#################################################

eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(False, False), duration=eval_env_steps_limit)) # set limit to m steps in the environment


#################################################
# Agent
#################################################

GANN_instance = pygad.gann.GANN(num_solutions=2,
                                num_neurons_input=19,
                                num_neurons_hidden_layers=fc_layer_params,
                                num_neurons_output=4,
                                hidden_activations=activations,
                                output_activation="sigmoid")

#population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)


#################################################
# Training and Evaluation functions
#################################################

data_plotter = Plotter()
avg_rewards = np.empty((0,2))

def fitness_func(solution, sol_idx):
    global GANN_instance

    total_reward = 0
    print('Evaluation chromosome:', sol_idx)
    start = time.time()
    time_step = eval_tf_env.reset()
    while not time_step.is_last():
      action_step = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                      data_inputs=time_step.observation,
                                      problem_type="regression")
      time_step = eval_tf_env.step(tf.convert_to_tensor([action_step[0]], dtype=tf.float32))
      total_reward += float(time_step.reward)
    end = time.time()
    print('Control loop timing for 1 timestep [s]:', (end-start)/eval_env_steps_limit)
    
    solution_fitness = total_reward
    return solution_fitness

def callback_generation(ga_instance):
    global GANN_instance, data_plotter, avg_rewards

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solutions_fitness[-1]))

    avg_rewards = np.concatenate((avg_rewards, [[ga_instance.generations_completed, ga_instance.best_solutions_fitness[-1]]]), axis=0)
    data_plotter.update_eval_reward(ga_instance.best_solutions_fitness[-1], 1)
    time.sleep(0.1)


ga_instance = pygad.load(save_path+"/pygad_GA")

best_sol = ga_instance.best_solutions[np.argmax(ga_instance.best_solutions_fitness)]

population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=[best_sol, best_sol])
GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)


total_reward=0
time_step = eval_tf_env.reset()
while not time_step.is_last():
  action_step = pygad.nn.predict(last_layer=GANN_instance.population_networks[0],
                                  data_inputs=time_step.observation,
                                  problem_type="regression")
  time_step = eval_tf_env.step(tf.convert_to_tensor([action_step[0]], dtype=tf.float32))
  total_reward += float(time_step.reward)

print(total_reward)