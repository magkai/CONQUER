import sys
import json
import pickle
import tensorflow as tf 
from transformers import BertTokenizer, TFBertModel
sys.path.append("../utils")
import utils as ut

"""Encode questions and actions with pre-trained BERT model"""

#load all questions (this needs to be done for dev and test data accordingly)
with open("../data/train_data/all_questions_trainset.json", "r") as questionFile:
    train_questions = json.load(questionFile)

#load all paths for each startpoint per question
with open("../data/train_data/contextPaths_trainset.json", "r") as afile:
    train_paths = json.load(afile)

with open("../data/dev_data/all_questions_devset.json", "r") as questionFile:
    dev_questions = json.load(questionFile)

#load all paths for each startpoint per question
with open("../data/dev_data/contextPaths_devset.json", "r") as afile:
    dev_paths = json.load(afile)

with open("../data/test_data/all_questions_testset.json", "r") as questionFile:
    test_questions = json.load(questionFile)

#load all paths for each startpoint per question
with open("../data/test_data/contextPaths_testset.json", "r") as afile:
    test_paths = json.load(afile)


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_encoding = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)

#get pre-trained BERT embeddings for question
def encodeQuestion(question):
    tokenized_input =  bert_tokenizer(question, return_tensors="tf", padding=True) 
    encoded_input = bert_encoding(tokenized_input, output_hidden_states=True)
    #take average over all hidden layers
    all_layers = [ encoded_input.hidden_states[l] for l in range(1,13)]
    encoder_layer = tf.concat(all_layers, 1)
    pooled_output = tf.reduce_mean(encoder_layer, axis=1)

    return pooled_output


#get pre-trained BERT embeddings for actions
def encodeActions(actions):
    try:
        tokenized_actions =  bert_tokenizer(actions, return_tensors="tf", padding=True, truncation=True, max_length=50) 
    except Exception as e:
        print("error: ", e) 
        print("actions not working: ", actions)
        return None
 
    encoded_actions = bert_encoding(tokenized_actions, output_hidden_states=True)
    #take average overall all hidden layers
    all_layers = [ encoded_actions.hidden_states[l] for l in range(1,13)]
    encoder_layer = tf.concat(all_layers, 1)
    pooled_output = tf.reduce_mean(encoder_layer, axis=1)
 
    return pooled_output


#get node labels for each path (this can be adapted to also include start (paths) and endpoint as action)
def getActionLabels(paths):
    action_labels = dict()
    for key in paths.keys():
        action_labels[key] = []
        actions = paths[key]
        for a in actions:
            p_labels = ""
            #use this if startpoint should be included in action:
            #p_labels = ut.getLabel(a[0]) + " "
            for aId in a[1]:
                p_labels += ut.getLabel(aId) + " "
            #use this if endpoint should be included in action 
            #p_labels += ut.getLabel(a[2])
            action_labels[key].append(p_labels)
    return action_labels


#get all action embeddings for paths in the dataset
def getActionEncodings(action_labels):
    all_encoded_paths = dict()
    action_nbrs = dict()
    for start in action_labels.keys():
        #store how many paths are available per startpoint
        action_nbrs[start] = len(action_labels[start])
        if action_nbrs[start] == 0:
            continue
        first = True
        encoded_paths = None
        j = -1
        #encode paths batchwise
        for i in range(action_nbrs[start]):   
            j+=1
            if j == 64:
                if first:
                    encoded_paths = encodeActions(action_labels[start][i-j:i+1])
                    if encoded_paths is None:
                        j = -1
                        continue
                    first = False
                else:
                    encoded_actions = encodeActions(action_labels[start][i-j:i+1])
                    if encoded_actions is None:
                        j = -1
                        continue
                    encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions],axis=0)
                j = -1
        encoded_actions = encodeActions(action_labels[start][i-j:i+1])
        if encoded_actions is None and encoded_paths is None:
            continue
        if not encoded_actions is None:
            if first:
                encoded_paths = encoded_actions
            else:
                encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions],axis=0)
        #pad all paths to length of 1000
        if len(encoded_paths) < 1000:
            zeros = tf.zeros((1000-action_nbrs[start], 768))
            encoded_paths = tf.keras.layers.concatenate([encoded_paths, zeros],axis=0)
        all_encoded_paths[start] = encoded_paths

    return all_encoded_paths, action_nbrs


#encode questions
encoded_train_questions = dict()
encoded_dev_questions = dict()
encoded_test_questions = dict()

for qId in train_questions.keys():
    encoded_train_questions[qId] = encodeQuestion(train_questions[qId])

for qId in dev_questions.keys():
    encoded_dev_questions[qId] = encodeQuestion(dev_questions[qId])

for qId in test_questions.keys():
    encoded_test_questions[qId] = encodeQuestion(test_questions[qId])

with open("../data/train_data/encoded_questions_trainset.pickle", "wb") as q_file:
    pickle.dump(encoded_train_questions, q_file)

with open("../data/dev_data/encoded_questions_devset.pickle", "wb") as q_file:
    pickle.dump(encoded_dev_questions, q_file)

with open("../data/test_data/encoded_questions_testset.pickle", "wb") as q_file:
    pickle.dump(encoded_test_questions, q_file)

print("question encoding done")

#get action labels (needed for action encoding)
train_action_labels = getActionLabels(train_paths)
dev_action_labels = getActionLabels(dev_paths)
test_action_labels = getActionLabels(test_paths)

with open("../data/train_data/action_labels_trainset.json", "w") as qfile:
   json.dump(train_action_labels, qfile)

with open("../data/dev_data/action_labels_devset.json", "w") as qfile:
   json.dump(dev_action_labels, qfile)

with open("../data/test_data/action_labels_testset.json", "w") as qfile:
   json.dump(test_action_labels, qfile)

#encode actions batchwise
encoded_train_paths, train_action_nbrs = getActionEncodings(train_action_labels)
encoded_dev_paths, dev_action_nbrs = getActionEncodings(dev_action_labels) 
encoded_test_paths, test_action_nbrs = getActionEncodings(test_action_labels)

with open("../data/train_data/encoded_paths_trainset.pickle", "wb") as a_file:
   pickle.dump(encoded_train_paths, a_file)

with open("../data/train_data/action_numbers_trainset.json", "w") as nbr_file:
   json.dump(train_action_nbrs, nbr_file)

with open("../data/dev_data/encoded_paths_devset.pickle", "wb") as a_file:
   pickle.dump(encoded_dev_paths, a_file)

with open("../data/dev_data/action_numbers_devset.json", "w") as nbr_file:
   json.dump(dev_action_nbrs, nbr_file)

with open("../data/test_data/encoded_paths_testset.pickle", "wb") as a_file:
   pickle.dump(encoded_test_paths, a_file)

with open("../data/test_data/action_numbers_testset.json", "w") as nbr_file:
   json.dump(test_action_nbrs, nbr_file)

print("action encoding done")



