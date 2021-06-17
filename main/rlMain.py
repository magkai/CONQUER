from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys

"""CONQUER main training file"""

#Set seed values everywhere to make results reproducible
with open(sys.argv[1], "r") as config_file:
  config = json.load(config_file)
# Set a seed value
seed_value= config["seed"] # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)# 4. Set `tensorflow` pseudo-random generator at a fixed value
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
import tensorflow as tf
tf.random.set_seed(seed_value)


from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.agents import ReinforceAgent
from tf_agents.trajectories.time_step import StepType

import pickle

from rlEnvironment import RLEnvironment
from policyNetwork import KGActionDistNet


tf.compat.v1.enable_v2_behavior()
train_step_counter = tf.compat.v2.Variable(0)

#get parameters from config file 
entropy_const = config["entropy_const"]
learning_rate = config["learning_rate"]
num_episodes = config["num_episodes"]
num_rollouts = config["num_rollouts"]
num_epochs = config["num_epochs"]
#whether the reformulation predictor is used
ref_prediction = config["ref_prediction"]
#whether the user model is noisy (=real)
real_user = config["real_user"]
#whether fractional reward directly coming from the ref predictor should be used
fractional_reward = config["fractional_reward"]

#whether a pretrained model should be used
if "pretrained" in config.keys():
  pretrained = config["pretrained"]
else: 
  pretrained = False
#whether negative reward should be used instead reward = 0
if "negative_reward" in config.keys():
  alt_reward = config["negative_reward"]
else: 
  alt_reward = True

#reinforce requires a BoundedArrayspec, selected actions are numeric values sampled from categorical distribution
action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=999, name='action')
observation_spec = array_spec.ArraySpec(shape=(config["observation_spec_shape_x"],config["observation_spec_shape_y"]), dtype=np.float32, name='observation')

#list with all question ids
with open(config["question_list"], "r") as conv_file:
  question_list = json.load(conv_file)

#the startpoints we have determined upfront for each question in the trainset
with open(config["starts_per_question"], "r") as start_file:
  starts_per_question = json.load(start_file)

#paths per startpoint
with open(config["paths"], "r") as path_file:
  paths = json.load(path_file)

#all answers in trainset
with open(config["answers"], "r") as answerFile:
  answers = json.load(answerFile)

#for each question get next info need
with open(config["next_questions"], "r") as qfile:
  next_questions = json.load(qfile)

#for each question get the next reformulation from ConvRef
with open(config["next_reformulations"], "r") as qfile:
  next_reformulations = json.load(qfile)

#indices of question and startpoint ids
with open(config["q_start_indices"], "r") as qfile:
  q_start_indices =  json.load(qfile)

#when ref predictor is used, predictions for each question and its follow up info need
if "next_info_predictions" in config.keys():
  with open(config["next_info_predictions"], "r") as qfile:
    next_info_predictions = json.load(qfile)
else:
  next_info_predictions = None

#when ref predictor is used: predictions for each question and its next reformulation (from ConvRef)
if "ref_predictions" in config.keys():
  with open(config["ref_predictions"], "r") as qfile:
    ref_predictions = json.load(qfile)
else:
  ref_predictions = None

#if fractional reward should be used: get directly ref predictor probabilities for how likely question pair have same info need
if "ref_fraction" in config.keys():
  with open(config["ref_fraction"], "r") as frac_file:
    ref_fraction = json.load(frac_file)
else:
  ref_fraction = None

#if fractional reward should be used: get directly ref predictor probabilities for how likely question pair have different info need
if "next_info_fraction" in config.keys():
  with open(config["next_info_fraction"], "r") as q_file:
    next_info_fraction = json.load(q_file)
else:
  next_info_fraction = None

#BERT embedded history
if "encoded_history" in config.keys():
  with open(config["encoded_history"], "rb") as q_file:
    encoded_history = pickle.load(q_file)
else:
  encoded_history = None

#BERT embedded questions
with open(config["encoded_questions"], "rb") as q_file:
  encoded_questions = pickle.load(q_file)

#BERT embedded actions
with open(config["encoded_actions"], "rb") as a_file:
  encoded_actions = pickle.load(a_file)

#number of actions available per startpoint
with open(config["action_nbrs"], "r") as nbr_file:
  action_nbrs = json.load(nbr_file)


#initialize the environment 
kgEnv = RLEnvironment(observation_spec, action_spec, encoded_history, encoded_questions, question_list,
starts_per_question, q_start_indices, encoded_actions, action_nbrs,  answers, paths, ref_prediction, real_user, 
fractional_reward, next_info_predictions, ref_predictions, next_info_fraction, ref_fraction, next_questions, next_reformulations, alt_reward)


train_env = tf_py_environment.TFPyEnvironment(kgEnv)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

#initialize the policy network
actor_network = KGActionDistNet( 
  seed_value,
  train_env.observation_spec(), 
  train_env.action_spec())

#initialize the agent
rfAgent = ReinforceAgent(
     train_env.time_step_spec(), tensor_spec.from_spec(action_spec), actor_network, optimizer, entropy_regularization=entropy_const, train_step_counter=train_step_counter
)
rfAgent.initialize()

#use a sampling policy
collect_policy = rfAgent.collect_policy


def collect_episodes_with_rollouts(environment, policy, num_episodes, num_rollouts):
  episode_counter = 0
  episode_return = 0.0
 
  while episode_counter < num_episodes:

    environment.set_is_rollout(False)
    #we are moving only one step each time
    time_step = environment.reset()

    if environment.is_final_observation():
      avg_return = episode_return / ((episode_counter+1)*num_rollouts)
      print("final obs")
      return avg_return
    #get back an action given our current state
    action_step = policy.action(time_step, seed=seed_value)
    #do next step on the environment
    next_time_step = environment.step(action_step.action)
    #collect the reward
    episode_return += next_time_step.reward
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    #store collected experience
    replay_buffer.add_batch(traj)
    #get action distribution from policy network
    distribution = actor_network.get_distribution()
    #sample numrollout-1 additional actions
    selectedActions =  tf.nest.map_structure(
        lambda d: d.sample((num_rollouts-1), seed=seed_value),
        distribution)
    #print("selected Actions: ", selectedActions)
   
    environment.set_is_rollout(True)
    #get from environment potential new state when alternative action is chosen
    for selAction in selectedActions:
      new_policy_step = action_step._replace(action=selAction)
      next_time_step = environment.step(selAction)
      episode_return += next_time_step.reward
      traj = trajectory.from_transition(time_step, new_policy_step, next_time_step)
      #store additional experience
      replay_buffer.add_batch(traj)

    episode_counter += 1
  #calculate average reward
  avg_return = episode_return / (num_episodes*num_rollouts)
  return avg_return


#buffer to store collected experience
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
     data_spec=collect_policy.collect_data_spec,
     batch_size= train_env.batch_size,
     max_length=5000)

#create checkpoint for weights of the policy network
checkpoint = tf.train.Checkpoint(actor_net=actor_network)

if pretrained:
  checkpoint.restore(config["pretrained_path"])


#main training loop
for j in range(num_epochs):
  kgEnv.reset_env()
  i = -1
  while True:
    i += 1
    #collect experience with additional rollouts
    average_return = collect_episodes_with_rollouts(kgEnv, collect_policy,num_episodes, num_rollouts)
    experience = replay_buffer.gather_all()
    if i == 0:
      print("weights: ", actor_network.trainable_weights, flush=True)
    #calculate loss
    train_loss = rfAgent.train(experience)
    if i % 100 == 0:
      print("iteration: ", i, flush=True)
      print("loss: ", train_loss.loss, flush=True)
      print("avg return: ", average_return, flush=True)
    replay_buffer.clear()

    if kgEnv.is_final_observation():
      break
  #save checkpoints for each epoch
  checkpoint.save(config["checkpoint_path"] + "-seed-"+ str(config["seed"]) + "/ckpt")
 

print("trained weights: ", actor_network.trainable_weights, flush=True)