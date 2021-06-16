
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

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
import tensorflow as tf
tf.random.set_seed(seed_value)


from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.agents import ReinforceAgent
from tf_agents.trajectories import time_step as ts


import pickle
import operator


from policyNetwork import KGActionDistNet
sys.path.append("../utils")
import utils as ut

tf.compat.v1.enable_v2_behavior()
train_step_counter = tf.compat.v2.Variable(0)
learning_rate = 1e-3


#load data from config file
filename = config["filename"]

#pre-computed context entities (=startpoints) for questions from testset/devset
with open(config["startpoints"], "r") as start_file:
    startpoints = json.load(start_file)

#available paths for each starting point
with open(config["contextPaths"], "r") as path_file:
    contextPaths = json.load(path_file)

#KG node labels
with open(config["labels_dict"]) as labelFile:
    labels_dict = json.load(labelFile)

#BERT embedded conversation context
if "bert_history" in config.keys(): 
    with open(config["bert_history"], "rb") as q_file:
        bert_history = pickle.load(q_file)
else:
    bert_history = None

#BERT embedded questions
with open(config["bert_questions"], "rb") as q_file:
    bert_questions = pickle.load(q_file)

#BERT embedded actions
with open(config["bert_actions"], "rb") as a_file:
    bert_actions = pickle.load(a_file)

#number of actions available for a given startpoint
with open(config["action_nbrs"], "r") as nbr_file:
    action_nbrs = json.load(nbr_file)

#aggregation type for final ranking
if "agg_type" in config.keys():
  agg_type = config["agg_type"]
else: 
  agg_type = "add"


#number of actions sampled per agent
nbr_sample_actions = config["nbr_sample_actions"]


action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=999, name='action')
observation_spec = array_spec.ArraySpec(shape=(config["observation_spec_shape_x"],config["observation_spec_shape_y"]), dtype=np.float32, name='observation')

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

#initialize policy network
actor_network = KGActionDistNet( 
    seed_value,
  tensor_spec.from_spec(observation_spec), 
  tensor_spec.from_spec(action_spec))


#restore trained policy network
checkpoint = tf.train.Checkpoint(actor_net=actor_network)
checkpoint.restore(config["checkpoint_path"] +  "-seed-"+ str(config["seed"])  + "/ckpt-" + str(config["checkpoint_nbr"])) 


rfAgent = ReinforceAgent(
    tensor_spec.from_spec(ts.time_step_spec(observation_spec)), tensor_spec.from_spec(action_spec), actor_network, optimizer, train_step_counter=train_step_counter
)
rfAgent.initialize()

#get greedy eval policy
eval_policy = rfAgent.policy

#check for existential question
def isExistential(question_start):
    existential_keywords	= ['is', 'are', 'was', 'were', 'am', 'be', 'being', 'been', 'did', 'do', 'does', 'done', 'doing', 'has', 'have', 'had', 'having']
    if question_start in existential_keywords:
        return True
    return False

#calculate P@1
def getPrecisionAt1(answers, goldanswers):
    goldanswers_lower = [ga.lower() for ga in goldanswers]
    for answer in answers:
        if answer[-1] > 1:
            return 0.0
        if answer[0].lower() in goldanswers_lower:
            return 1.0
   
    return 0.0

#calculate Hit@5
def getHitsAt5(answers, goldanswers):
    goldanswers_lower = [ga.lower() for ga in goldanswers]
    for answer in answers: 
        if answer[-1] > 5:
            return 0.0
        if answer[0].lower() in goldanswers_lower:
            return 1.0
    return 0.0

#calculate MRR
def getMRR(answers, goldanswers):
    goldanswers_lower = [ga.lower() for ga in goldanswers]
    i = 0
    for answer in answers:
      #  if i == cutoff:
       #     return 0.0
        if answer[0].lower() in goldanswers_lower:
            return 1.0/answer[-1]
        i+=1
    return 0.0



def call_rl(timesteps, start_ids):
    """apply trained policy network to get top answers (performed in parallel for all context entities/agents"""
    answers = dict()
    #perform one step with evaluation policy
    action_step = eval_policy.action(timesteps) 
    all_actions = np.arange(1000)
    all_actions = tf.expand_dims(all_actions, axis=1)
    #get action distribution from policy network
    #we need entire distribution over all action since we want to get top k actions (not only top-1)
    distribution = actor_network.get_distribution()
    #calculate probability scores and sample the top-k actions
    log_probability_scores = distribution.log_prob(all_actions)
    log_probability_scores = tf.transpose(log_probability_scores)
    top_log_scores, topActions = tf.math.top_k(log_probability_scores,nbr_sample_actions)
    #get respective answers by following path described by the selected action
    for i in range(len(start_ids)):
        for j in range(len(topActions[i])):
            if j == 0:
                answers[start_ids[i]] = []
            if not start_ids[i] in contextPaths.keys():
                answers[start_ids[i]].append("")
                continue
            paths = contextPaths[start_ids[i]]
            if topActions[i][j].numpy() >= len(paths):
                answers[start_ids[i]].append("")
                continue
            answerpath = paths[topActions[i][j]]
            if len(answerpath) < 3:
                answers[start_ids[i]].append("")
                continue
            answer = answerpath[2]
            answers[start_ids[i]].append(answer)
          
    return (answers, top_log_scores.numpy())
    
def create_observation(startId, encoded_history, encoded_question):
    """create input to policy network"""

    if not startId in bert_actions.keys():
        #print("Warning: No available actions for starting point with id: ", startId)
        return None
   
    encoded_actions = bert_actions[startId]
    action_nbr = action_nbrs[startId]
    
    mask = tf.ones(action_nbr)
    if encoded_history is None:
        zeros = tf.zeros((1001-action_nbr))
    else:
        zeros = tf.zeros((1002-action_nbr))
    mask = tf.keras.layers.concatenate([mask, zeros], axis=0)
    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)#[1,1001,1]
    
    if encoded_history is None:
        observation = tf.keras.layers.concatenate([encoded_question, encoded_actions],axis=0) #[1001, 768]
    else:
        observation = tf.keras.layers.concatenate([encoded_history, encoded_question, encoded_actions],axis=0) #[1002, 768]
        
 
    observation = tf.expand_dims(observation, 0) #[1, 1001, 768]
    observation =  tf.keras.layers.concatenate([observation, mask], axis=2) #[1,1001,769]
    tf.dtypes.cast(observation, tf.float32)

    return observation


def findAnswer(currentid):
    """main method for retrieving answers
       for all available context entities get predictions from trained policy network (= parallel agents walk) 
       and take the top-k answers per agent"""

    if not currentid in startpoints.keys():
        return []
    currentStarts = startpoints[currentid]  
    if len(currentStarts) == 0:
        return []
    turn = int(currentid.split("-")[1])
   
    timeSteps = None
    observations = None
    startIds = []
    #prepare input for policy network for each start point
    for startNode in currentStarts:
        if bert_history:
            observation = create_observation(startNode,bert_history[currentid],bert_questions[currentid])
        else:
            observation = create_observation(startNode,None,bert_questions[currentid])
        if observation is None:
            continue
        if observations is None:
            observations = observation
        else:
            observations = np.concatenate((observations, observation))
        startIds.append(startNode)
   
    if not observations is None:
        timeSteps = ts.restart(observations, batch_size=observations.shape[0])
  
    if not timeSteps is None:
        #get predictions from policy network
        answers, log_probs = call_rl(timeSteps, startIds)
    else:
        return []
    
    i = 0
    answer_scores = dict() 
    majority_answer_scores = dict()
    maxmajo_answer_scores = dict()
    additive_answer_scores = dict() #default

    for sId in startIds:
        for j in range(len(log_probs[i])):
            score = np.exp(log_probs[i][j])
   
            curr_answer = answers[sId][j]
            if curr_answer == "":
              #  print("empty answer for startid: ", sId[0], "with log prob: ", log_probs[i][j])
                continue
            #for additive aggregation type, take sum of scores when several agents have same answer entity
            if agg_type == "add":
                if curr_answer in additive_answer_scores.keys():
                    additive_answer_scores[curr_answer] += score
                else: 
                    additive_answer_scores[curr_answer] = score
            #take maximal answer score
            elif agg_type == "max":
                if curr_answer in answer_scores.keys():
                    if score > answer_scores[curr_answer]:
                        answer_scores[curr_answer] = score
                else:
                    answer_scores[curr_answer] = score
            #store both: score and count of how many agent arrive at entity
            elif agg_type == "majo":
                if curr_answer in majority_answer_scores.keys():
                    if score > majority_answer_scores[curr_answer][0]:
                        majority_answer_scores[curr_answer][0] = score
                    majority_answer_scores[curr_answer][1] += 1
                else: 
                    majority_answer_scores[curr_answer] = [score, 1]
            elif agg_type == "maxmajo":
                if curr_answer in maxmajo_answer_scores.keys():
                    if score > maxmajo_answer_scores[curr_answer][0]:
                        maxmajo_answer_scores[curr_answer][0] = score
                    maxmajo_answer_scores[curr_answer][1] += 1
                else: 
                    maxmajo_answer_scores[curr_answer] = [score, 1]
           
        i +=1

    if agg_type == "add":
        return sorted(additive_answer_scores.items(), key=lambda item: item[1], reverse=True)
    elif agg_type == "max":
        return sorted(answer_scores.items(), key=lambda item: item[1], reverse=True)
    #first use count as sorting criterion, then score
    elif agg_type == "majo":
        return [(y,majority_answer_scores[y][1], majority_answer_scores[y][0]) for y in sorted(majority_answer_scores, key=lambda x: (majority_answer_scores[x][1], majority_answer_scores[x][0]), reverse=True)]
    #use score as first sorting criterion, count as second
    elif agg_type == "maxmajo":
        return [(y,maxmajo_answer_scores[y][0], maxmajo_answer_scores[y][1]) for y in sorted(maxmajo_answer_scores, key=lambda x: (maxmajo_answer_scores[x][0], maxmajo_answer_scores[x][1]), reverse=True)]

    additive_answer_scores = sorted(additive_answer_scores.items(), key=lambda item: item[1], reverse=True)
    return additive_answer_scores
        
        

def calculateAnswerRanks(answer_scores):
    """calculate ranks, several results can share the same rank if they have the same score"""
    if len(answer_scores) == 0:
        return []
    rank = 0
    same_ranked = 0
    prev_score = ["", ""]
   
    for score in answer_scores:
        if score[1] == prev_score[1]:
            #for these aggregation types we need to check two scores
            if agg_type == "majo" or agg_type == "maxmajo":
                if score[2] == prev_score[2]:
                    same_ranked +=1 
                else:
                    rank += (1 + same_ranked)
                    same_ranked = 0
            else:
                same_ranked +=1 
        else:
            rank += (1 + same_ranked)
            same_ranked = 0
        score.append(rank)
        prev_score = score
    
    return answer_scores


def formatAnswer(answer):
    if len(answer) == 0:
        return answer
    best_answer = answer
    if answer[0] == "Q" and "-" in answer:
        best_answer = answer.split("-") [0]
    elif ut.is_timestamp(answer):
        best_answer = ut.convertTimestamp(answer)
 
    return best_answer

if __name__ == '__main__':
   
    with open(config["filename"] + "-seed-"+ str(config["seed"]) + ".csv", "w") as resFile:
   
        header = "CONV_ID,QUESTION,ANSWER,GOLD_ANSWER,PRECISION@1, HITS@5, MRR" + "\n"
        resFile.write(header)
        with open("../data/ConvRef/ConvRef_testset.json") as json_file:
            data = json.load(json_file)
           
            avg_prec = 0.0
            avg_hits_5 = 0.0
            avg_mrr = 0.0
            prec = []
            hits_5 = []
            mrr = []
            question_count = 0
            existential_count = 0
            ref_count = 0
            #stores how many questions are answered after x reformulations
            answer_tries = []
            #store how many question are answered in total
            total_answered = 0
            result_dict = dict()
            
            for i in range(8):
                answer_tries.append(0)
            #go over each conversation
            for conv in data:
                #go over each question in current conversation
                for question_info in conv['questions']:
                    policy_state = eval_policy.get_initial_state(batch_size=1)
                    prec = []
                    hits_5 = []
                    mrr = []
                    question_count += 1
                    question_start = question_info['question'].split(" ")[0].lower()
                    question_id = question_info['question_id']
                    question = question_info["question"]
                    #get gold answers for metric calculations
                    gold_answers = ut.getGoldAnswers(question_info["gold_answer"])
                    #find answer for current question
                    answer_scores = findAnswer(question_id)
                    answer_scores = [list(a) for a in answer_scores]             
                      
                    for answer in answer_scores:
                        if agg_type == "majo":
                            answer[2] = format(answer[2],  '.3f')
                        else:
                            answer[1] = format(answer[1],  '.3f')
                        answer[0] = formatAnswer(answer[0]) 
                    calculateAnswerRanks(answer_scores)

                    if isExistential(question_start):
                        answer_scores = [["Yes", 1.0 , 1], ["No", 0.5 ,2]]
                
                    result_dict[question_id] = answer_scores
                    #calculate metrics
                    prec.append(getPrecisionAt1(answer_scores, gold_answers))
                    hits_5.append(getHitsAt5(answer_scores, gold_answers))
                    mrr.append(getMRR(answer_scores,gold_answers))
                    #write results to csv file
                    if answer_scores == []:
                        resFile.write(question_id + "," + question + "," + str([]) + "," + str(gold_answers) + "," + str(prec[0]) + "," + str(hits_5[0]) + "," + str(mrr[0]) + "\n")
                    else:
                        resFile.write(question_id + "," + question + "," + answer_scores[0][0] + "," + str(gold_answers) + "," + str(prec[0]) + "," + str(hits_5[0]) + "," + str(mrr[0]) + "\n")
                    resFile.flush()
              
                    #check if correct answer was retrieved and update stats
                    if prec[0] == 1.0:
                        avg_prec+= prec[0]
                        avg_hits_5 += hits_5[0]
                        avg_mrr += mrr[0]
                        answer_tries[0] += 1
                        total_answered += 1
                        continue
                    #if question was not answered correctly: next reformulation is triggered
                    for i in range(len(question_info["reformulations"])):
                        ref_count += 1
                        ref_id = question_info["reformulations"][i]["ref_id"]
                        question_start =question_info["reformulations"][i]["reformulation"].split(" ")[0].lower()
                        #find answer for reformulation
                        answer_scores = findAnswer(ref_id)
                        answer_scores = [list(a) for a in answer_scores]
                        
                        for answer in answer_scores:
                            if agg_type == "majo":
                                answer[2] = format(answer[2],  '.3f')
                            else:
                                answer[1] = format(answer[1],  '.3f')
                            answer[0] = formatAnswer(answer[0]) 
                        calculateAnswerRanks(answer_scores)

                        if isExistential(question_start):
                            # always Yes:
                            answer_scores = [["Yes", 1.0 , 1], ["No", 0.5 ,2]]

                        result_dict[ref_id] = answer_scores
                        #calculate metrics
                        prec.append(getPrecisionAt1(answer_scores, gold_answers))
                        hits_5.append(getHitsAt5(answer_scores, gold_answers))
                        mrr.append(getMRR(answer_scores,gold_answers))
                        #write results to csv file
                        if answer_scores == []:
                            resFile.write(ref_id + "," + question_info["reformulations"][i]["reformulation"] + "," + str([]) + "," + str(gold_answers) + "," + str(prec[i+1]) + "," + str(hits_5[i+1]) + "," + str(mrr[i+1]) + "\n")
                        else:
                            resFile.write(ref_id + "," + question_info["reformulations"][i]["reformulation"] + "," + answer_scores[0][0] + "," + str(gold_answers) + "," + str(prec[i+1]) + "," + str(hits_5[i+1]) + "," + str(mrr[i+1]) + "\n")
                        resFile.flush()
                  
                        #we can stop to trigger refs if correct answer has been found
                        if prec[i+1] == 1.0:
                            if i > 4:
                               print("ref_id answered correctly after forth try: ", ref_id)
                            answer_tries[i+1] += 1
                            total_answered += 1
                            break
                       
                    #for final metric: sort based on mrr 
                    index, _ = max(enumerate(mrr), key=operator.itemgetter(1))
                    avg_prec += prec[index]
                    avg_hits_5 += hits_5[index]
                    avg_mrr += mrr[index]
                   

            #write results to file   
            resFile.write("AVG results:, , , ," + str(format(avg_prec/question_count, '.3f')) + "," +  str(format(avg_hits_5/question_count, '.3f')) + "," +  str(format(avg_mrr/question_count, '.3f')) + "\n" )
            resFile.write("total number of information needs: "+ str(question_count)+ "\n" )
            resFile.write("total number of reformulations triggered: "+ str(ref_count)+ "\n" )
            resFile.write("total number of correctly answered questions: "+ str(total_answered)+ "\n" )
            resFile.write("number of questions answered correctly on first attempt: "+ str(answer_tries[0])+ "\n" )
            resFile.write("number of questions answered correctly on second attempt: "+ str(answer_tries[1])+ "\n" )
            resFile.write("number of questions answered correctly on third attempt: "+ str(answer_tries[2])+ "\n" )
            resFile.write("number of questions answered correctly on forth attempt: "+ str(answer_tries[3])+ "\n" )
            resFile.write("number of questions answered correctly on fifth attempt: "+ str(answer_tries[4])+ "\n" )

            
          