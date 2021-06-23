import json
import sys
sys.path.append("../utils")
import utils as ut

"""Prepare startPoints for training"""
####only used for TRAINING data not for dev and test data####

#load retrieved startpoints
with open("../data/train_data/startPoints_trainset.json", "r") as s_file:
    startPoints = json.load(s_file)
#load corresponding paths (max 1000) for each start point
with open("../data/train_data/contextPaths_trainset.json", "r") as path_file:
    paths = json.load(path_file)
#load all gold answers
with open("../data/train_data/all_answers_trainset.json", "r") as answerFile:
    answers = json.load(answerFile)
#get a list of question ids
with open("../data/train_data/questionId_list_trainset.json", "r") as conv_file:
    qList = json.load(conv_file)


#propagates startpoints to reformulations/initial question with same intent to make best use of the available data
def propagateStartpoints():
    for key in startPoints.keys():
        startPoints[key] = list(set(startPoints[key]))
        #check if question was a reformulation
        if key.count("-") == 2:
            #get corresponding main id
            main_key = "-".join(key.split("-", 2)[:2])
            #potentially add startpoint available for the initial question
            if main_key in startPoints.keys():
                if not len(startPoints[main_key]) == 0:
                    for mid in startPoints[main_key]:
                        if not mid in startPoints[key]:
                            startPoints[key].append(mid)
            
            for i in range(5):
                #go over other reformulations and potentially add a startpoint
                cId = main_key + "-" + str(i)
                if key == cId:
                    continue
                if cId in startPoints.keys():
                    if not len(startPoints[cId]) == 0:
                        for rid in startPoints[cId]:
                            if not rid in startPoints[key]:  
                                startPoints[key].append(rid)
        #initial question
        else:
            #go over each reformulation and potentially add another startpoint
            for i in range(5):
                cId = key + "-" + str(i)
                if cId in startPoints.keys():
                    if not len(startPoints[cId]) == 0:
                        for rid in startPoints[cId]:
                            if not rid in startPoints[key]:
                                startPoints[key].append(rid)
                               

#limits the startpoints to those from which the answer is reachable in one hop to improve training (otherwise agent could only have bad actions to choose from)
def getReachableStartpoints():
    number_of_training_samples = 0
    number_of_samples_answer_reachable = 0
    reachable_start_points = dict()
    for qid in startPoints.keys():
        for stp in startPoints[qid]:
            number_of_training_samples += 1
            for path in paths[stp]:
                pathend = path[2] 
                if ut.is_timestamp(pathend):
                    pathend = ut.convertTimestamp(pathend)
                #check if path leads to answer
                if pathend in answers[qid]:   
                    number_of_samples_answer_reachable += 1
                    #add startpoint to reachable ones
                    if qid in reachable_start_points.keys():
                        reachable_start_points[qid].append(stp)
                    else:
                        reachable_start_points[qid] = [stp]
                    #sufficient if answer can be found for one path from particular startpoint
                    break
    
    #print stats
    print("number of training samples: ", number_of_training_samples)
    print("number of training samples where answer reachable from startpoint: ", number_of_samples_answer_reachable)
    return reachable_start_points


#store indices for question and the respective startpoints for easier processing later
def createQuestionStartpointList(starts):
    q_start_indices = []
    for i in range(len(qList)):
        if qList[i] in starts.keys(): #this line means we ignore question ids for which we don't have any startpoints
        #   if not qList[i] in removeList:
            for j in range(len(starts[qList[i]])):
                q_start_indices.append([i, j])
    return q_start_indices


propagateStartpoints()

with open("../data/train_data/startpoints_propagated_trainset.json", "w") as s_file:
    json.dump(startPoints, s_file)


reachable_start_points = getReachableStartpoints()

with open("../data/train_data/reachable_startpoints_propagated_trainset.json", "w") as stp_file:
    json.dump(reachable_start_points, stp_file)


q_start_indices = createQuestionStartpointList(reachable_start_points)

with open("../data/train_data/question_start_indices_trainset.json", "w") as qfile:
   json.dump(q_start_indices, qfile)
  
