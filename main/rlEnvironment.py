from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import re
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

sys.path.append("../utils")
import utils as ut



"""CONQUER environment"""
class RLEnvironment(py_environment.PyEnvironment):

  
    def __init__(self, observation_spec, action_spec, all_history, all_questions, questionIds,  starts_per_question, q_start_indices,
     all_actions, action_nbrs, all_answers,  paths, ref_prediction, real_user, fractional_reward, next_info_predictions, ref_predictions,
      next_info_fraction, ref_fraction, next_questions, next_reformulations, alt_reward):
        """observation_spec: observation specification
            action spec: action specification
            all_history: history encodings
            all_questions: question encodings
            questionIds: list with all question ids
            starts_per_question: context entities per startpoint
            q_start_indices: indices for qid and context entity number
            all_actions: action encodings
            action_nbrs: number of action (=number of paths per context entity)
            all_answers: gold label answers (for user simulation)
            paths: KG paths from context entities
            ref_prediction: whether noisy reformulation predictor should be used
            real_user: whether noisy (=real) user  model should be used
            fractional_reward: whether reward based on the reformulation probability should be used
            next_info_predictions: predictions from ref predictor for next info needs 
            ref_predictions: ref predictions for reformulations
            next_info_fraction: probabilities for next info prediction 
            ref_fraction: probabilities for ref prediction  
            next_questions: next info need per question
            next_reformulations: next reformulation per question
            alt_reward: whether -1 should be used as negative reward"""

        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self.questionIds = questionIds
        self.all_answers = all_answers
        self.q_start_indices = q_start_indices
        self.all_questions = all_questions
        self.all_history = all_history
        self.all_actions = all_actions
        self.number_of_actions = action_nbrs

        self.starts_per_question = starts_per_question
        self.question_counter = 0
      
        self.paths = paths
        self.final_obs = False
        self._batch_size = 1
      
        self.ref_prediction = ref_prediction
        self.real_user = real_user
        self.fractional_reward = fractional_reward
        self.next_questions = next_questions
        self.next_reformulations = next_reformulations
        self.next_info_predictions = next_info_predictions
        self.ref_predictions = ref_predictions
        self.next_info_fraction = next_info_fraction
    
        self.ref_fraction = ref_fraction
        self.alt_reward = alt_reward
  
        super(RLEnvironment, self).__init__()
     

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def _empty_observation(self):
        return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype),
                                 self.observation_spec())

    def setTopActions(self, actions):
        self.topActions = actions


    def _get_observation(self):
        """Returns an observation."""
        #we want to go over each question, and for each question over each possible starting point for that question

        if self.question_counter == len(self.q_start_indices):
            print("end of training samples: empty observation returned")
            self.final_obs = True
            return self._empty_observation()

        #get next training ids for question and startpoints
        q_counter, start_counter = self.q_start_indices[self.question_counter] 
        self.qId = self.questionIds[q_counter]
     
        self.start_id = self.starts_per_question[self.qId][start_counter]
        self.question_counter+= 1

        #get pre-computed bert embeddings for the question
        encoded_question = self.all_questions[self.qId]

        #optional: get history embeddings
        if self.all_history:
            encoded_history = self.all_history[self.qId]
     
        #get action embeddings
        encoded_actions = self.all_actions[self.start_id]
        action_nbr = self.number_of_actions[self.start_id]
       
        mask = tf.ones(action_nbr)
        if self.all_history:
            zeros = tf.zeros((1002-action_nbr))
        else:
            zeros = tf.zeros((1001-action_nbr))
        mask = tf.keras.layers.concatenate([mask, zeros], axis=0)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1)#[1,1001,1] or [1,1002,1]
     
        #put them together as next observation for the policy network
        if self.all_history:
            observation = tf.keras.layers.concatenate([encoded_history, encoded_question, encoded_actions],axis=0) #[1002, 768]
        else:
            observation = tf.keras.layers.concatenate([encoded_question, encoded_actions],axis=0) #[1001, 768]
       
        observation = tf.expand_dims(observation, 0) #[1, 1001, 768] (or [1, 1002, 768])
        observation =  tf.keras.layers.concatenate([observation, mask], axis=2) #[1,1001,769] (or [1, 1002, 769])
        tf.dtypes.cast(observation, tf.float32)
 
        return observation

       
    def _reset(self):
        self._done = False
        obs = self._get_observation()
        if self.final_obs:
            print("final obs inside reset")
            return ts.termination(self._empty_observation(), [0.0])
        return ts.restart(obs, batch_size=self._batch_size)

  
    def is_final_observation(self):
        return self.final_obs

    def set_is_rollout(self, rollout):
        self.rollout = rollout

    def reset_env(self):
        self.final_obs = False
        self.question_counter = 0
        self.qId = ""
        self.start_id = ""
        self.topActions = []
        self.rollout = False
     

    def _apply_action(self, action):
        """Applies `action` to the Environment 
        and returns the corresponding reward depending on our respective setting (noisy/ideal ref pred, noisy/ideal user model)

        Args:
        action: A value conforming action_spec that will be taken as action in the
            environment.

        Returns:
        A float value that is the reward received by the environment.
        """

        answer = self.paths[self.start_id][action[0].numpy()][2]
        if ut.is_timestamp(answer):
            answer = ut.convertTimestamp(answer)
        
        #check whether the prediction from the reformulation predictor should be used or not
        if not self.ref_prediction:
            #check if a noisy ("real") user model is used or not
            if self.real_user:
                next_question = self._user_simulator(answer)
                #if user issues new info need -> assume prev. answer was correct, reward = 1
                if next_question == self.next_questions[self.qId]:    
                    return [1.0]
                else:
                    if self.alt_reward:
                        return [-1.0]
                    return [0.0]
            #ideal user/ideal ref. predictor (oracle version) = check gold labeled answer
            goldanswers = self.all_answers[self.qId]
            if answer in goldanswers:
                return [1.0]
            else:
                if self.alt_reward:
                    return [-1.0]
                return [0.0]
        #noisy ref prediction   
        else:
            #get next question from user simulator    
            next_question = self._user_simulator(answer)
            if next_question == self.next_questions[self.qId]: 
                #get pre-computed reformulation prediction for respective question and next question = new info need  
                current_ref_predict = self.next_info_predictions[self.qId]
                current_ref_fraction = None
                if not self.next_info_fraction is None:
                    current_ref_fraction = self.next_info_fraction[self.qId]
            #get pre-computed ref prediction for respective question and next question = reformulation
            else:                 
                current_ref_predict = self.ref_predictions[self.qId]
                current_ref_fraction = None
               
                if not self.ref_fraction is None:
                    current_ref_fraction = self.ref_fraction[self.qId]
            
            #check if ref prediction fraction should be used as reward signal
            if self.fractional_reward :
                if current_ref_predict == 1:
                    reward = 1.0-current_ref_fraction
                else:
                    reward = current_ref_fraction
        
                return [reward]
            
            #if reformulation then reward = 0
            reward = 1.0-current_ref_predict
            #check if neg. reward should be used as signal
            if self.alt_reward:
                if reward == 0.0:
                    reward = -1.0
         
            return [reward]
         
            

    def _user_simulator(self, answer):
        """Simulate user behavior: ideal case: always give a reformulation (cycle to collect refs in ConvRef) if presented answer is wrong, 
        noisy case: only give reformulation if another one is available in our ConvRef data"""
        next_question = ""
        goldanswers = self.all_answers[self.qId]
        #if answer correct: always issue new info need
        if answer in goldanswers:
            next_question = self.next_questions[self.qId]
            #print("correct answer")
        else:
            #in ideal case we always have a next reformulation
            if self.qId in self.next_reformulations.keys():
                next_question = self.next_reformulations[self.qId]
            #in noisy case: answer could be wrong but user continues with new information need
            else:
                next_question = self.next_questions[self.qId]
        return next_question


    
    def _step(self, action):

        reward = self._apply_action(action) 
        time_step = ts.termination(self._empty_observation(), reward)

        return time_step

    