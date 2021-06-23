
import numpy as np
import json
import pickle
import tensorflow as tf 

"""Get encodings for different variants using the conversation history"""


#load BERT encoded question
with open("../data/train_data/encoded_questions_trainset.pickle", "rb") as q_file:
  encoded_train_questions = pickle.load(q_file)

with open("../data/dev_data/encoded_questions_devset.pickle", "rb") as q_file:
  encoded_dev_questions = pickle.load(q_file)

with open("../data/test_data/encoded_questions_testset.pickle", "rb") as q_file:
  encoded_test_questions = pickle.load(q_file)

#get embeddings for the previous question
def getPreviousQuestionEncoding(qId, encoded_questions):
    if qId.count("-") == 2:
        qNum = qId[-3]
    else:
        qNum = qId[-1]
    if qNum == '0':
        return encoded_questions[qId]

    convid = qId.split("-")[0]
    prevqId = convid + "-" + str(int(qNum)-1)
    return encoded_questions[prevqId]

#get embeddings for first question
def getFirstQuestionEncoding(qId, encoded_questions):
    if qId.count("-") == 2:
        qNum = qId[-3]
    else:
        qNum = qId[-1]
    if qNum == '0':
        return encoded_questions[qId]
    convid = qId.split("-")[0]
    firstqId = convid + "-0" 
    return encoded_questions[firstqId]

#get average of embeddings over question/reformulations of previous intent
def getPreviousQuestionRefAverage(qId, encoded_questions):
    if qId.count("-") == 2:
        qNum = qId[-3]
    else:
        qNum = qId[-1]
    if qNum == '0':
        return getQuestionRefAverage(qId, encoded_questions)
    convid = qId.split("-")[0]
    prevqId = convid + "-" + str(int(qNum)-1)
    avg_prevQuestions =  getQuestionRefAverage(prevqId, encoded_questions)
    return avg_prevQuestions
    
#get average of embeddings over all reformulations for current intent
def getQuestionRefAverage(qId, encoded_questions):
    questions = encoded_questions[qId]
    for i in range(4):
        refId =  qId + "-" + str(i)
        if refId in encoded_questions.keys():
            questions = tf.keras.layers.concatenate([questions, encoded_questions[refId]], axis=0)
    avg_questions =  tf.reduce_mean(questions, axis=0)
    avg_questions = tf.expand_dims(avg_questions, 0)

    return avg_questions

#get average of embeddings over question/reformulations of first intent in conversation
def getFirstQuestionRefAverage(qId,encoded_questions):
    if qId.count("-") == 2:
        qNum = qId[-3]
    else:
        qNum = qId[-1]
    if qNum == '0':
        return getQuestionRefAverage(qId, encoded_questions)
    convid = qId.split("-")[0]
    firstqId = convid + "-0" 
    avg_firstQuestions =  getQuestionRefAverage(firstqId, encoded_questions)
    return avg_firstQuestions

#getfirst and previous questions embeddings
encoded_train_prevQuestions = dict()
encoded_train_firstQuestions = dict()
encoded_dev_prevQuestions = dict()
encoded_dev_firstQuestions = dict()
encoded_test_prevQuestions = dict()
encoded_test_firstQuestions = dict()
for qId in encoded_train_questions.keys():
    encoded_train_prevQuestions[qId] = getPreviousQuestionEncoding(qId, encoded_train_questions)
    encoded_train_firstQuestions[qId] = getFirstQuestionEncoding(qId, encoded_train_questions)
for qId in encoded_dev_questions.keys():
    encoded_dev_prevQuestions[qId] = getPreviousQuestionEncoding(qId, encoded_dev_questions)
    encoded_dev_firstQuestions[qId] = getFirstQuestionEncoding(qId, encoded_dev_questions)
for qId in encoded_test_questions.keys():
    encoded_test_prevQuestions[qId] = getPreviousQuestionEncoding(qId, encoded_test_questions)
    encoded_test_firstQuestions[qId] = getFirstQuestionEncoding(qId, encoded_test_questions)

with open("../data/train_data/encoded_prevQuestions_trainset.pickle", "wb") as q_file:
    pickle.dump(encoded_train_prevQuestions, q_file)

with open("../data/train_data/encoded_firstQuestions_trainset.pickle", "wb") as q_file:
    pickle.dump(encoded_train_firstQuestions, q_file)

with open("../data/dev_data/encoded_prevQuestions_devset.pickle", "wb") as q_file:
    pickle.dump(encoded_dev_prevQuestions, q_file)

with open("../data/dev_data/encoded_firstQuestions_devset.pickle", "wb") as q_file:
    pickle.dump(encoded_dev_firstQuestions, q_file)

with open("../data/test_data/encoded_prevQuestions_testset.pickle", "wb") as q_file:
    pickle.dump(encoded_test_prevQuestions, q_file)

with open("../data/test_data/encoded_firstQuestions_testset.pickle", "wb") as q_file:
    pickle.dump(encoded_test_firstQuestions, q_file)


#take average of embeddings first and previous question
encoded_train_firstQprevQAverage = dict()
encoded_dev_firstQprevQAverage = dict()
encoded_test_firstQprevQAverage = dict()
for qId in encoded_train_questions.keys():
    encoded_firstQ = encoded_train_firstQuestions[qId]
    encoded_prevQ = encoded_train_prevQuestions[qId]
    questions = tf.keras.layers.concatenate([encoded_firstQ, encoded_prevQ], axis=0)
    avg_questions =  tf.reduce_mean(questions, axis=0)
    avg_questions = tf.expand_dims(avg_questions, 0)
    encoded_train_firstQprevQAverage[qId] = avg_questions

for qId in encoded_dev_questions.keys():
    encoded_firstQ = encoded_dev_firstQuestions[qId]
    encoded_prevQ = encoded_dev_prevQuestions[qId]
    questions = tf.keras.layers.concatenate([encoded_firstQ, encoded_prevQ], axis=0)
    avg_questions =  tf.reduce_mean(questions, axis=0)
    avg_questions = tf.expand_dims(avg_questions, 0)
    encoded_dev_firstQprevQAverage[qId] = avg_questions

for qId in encoded_test_questions.keys():
    encoded_firstQ = encoded_test_firstQuestions[qId]
    encoded_prevQ = encoded_test_prevQuestions[qId]
    questions = tf.keras.layers.concatenate([encoded_firstQ, encoded_prevQ], axis=0)
    avg_questions =  tf.reduce_mean(questions, axis=0)
    avg_questions = tf.expand_dims(avg_questions, 0)
    encoded_test_firstQprevQAverage[qId] = avg_questions
    
with open("../data/train_data/encoded_firstQprevQAverage_trainset.pickle", "wb") as q_file:
    pickle.dump(encoded_train_firstQprevQAverage, q_file)

with open("../data/dev_data/encoded_firstQprevQAverage_devset.pickle", "wb") as q_file:
    pickle.dump(encoded_dev_firstQprevQAverage, q_file)

with open("../data/test_data/encoded_firstQprevQAverage_testset.pickle", "wb") as q_file:
    pickle.dump(encoded_test_firstQprevQAverage, q_file)

#get average of reformulation embeddings for first and previous questions
encoded_train_prevQuestion_RefAverage = dict()
encoded_train_firstQuestion_RefAverage = dict()
encoded_dev_prevQuestion_RefAverage = dict()
encoded_dev_firstQuestion_RefAverage = dict()
encoded_test_prevQuestion_RefAverage = dict()
encoded_test_firstQuestion_RefAverage = dict()
for qId in encoded_train_questions.keys():
    encoded_train_prevQuestion_RefAverage[qId] = getPreviousQuestionRefAverage(qId, encoded_train_questions)
    encoded_train_firstQuestion_RefAverage[qId] = getFirstQuestionRefAverage(qId, encoded_train_questions)
for qId in encoded_dev_questions.keys():
    encoded_dev_prevQuestion_RefAverage[qId] = getPreviousQuestionRefAverage(qId, encoded_dev_questions)
    encoded_dev_firstQuestion_RefAverage[qId] = getFirstQuestionRefAverage(qId, encoded_dev_questions)
for qId in encoded_test_questions.keys():
    encoded_test_prevQuestion_RefAverage[qId] = getPreviousQuestionRefAverage(qId, encoded_test_questions)
    encoded_test_firstQuestion_RefAverage[qId] = getFirstQuestionRefAverage(qId, encoded_test_questions)
 
with open("../data/train_data/encoded_firstQuestion_RefAverage_trainset.pickle", "wb") as q_file:
    pickle.dump(encoded_train_firstQuestion_RefAverage, q_file)

with open("../data/train_data/encoded_prevQuestion_RefAverage_trainset.pickle", "wb") as q_file:
    pickle.dump(encoded_train_prevQuestion_RefAverage, q_file)

with open("../data/dev_data/encoded_firstQuestion_RefAverage_devset.pickle", "wb") as q_file:
    pickle.dump(encoded_dev_firstQuestion_RefAverage, q_file)

with open("../data/dev_data/encoded_prevQuestion_RefAverage_devset.pickle", "wb") as q_file:
    pickle.dump(encoded_dev_prevQuestion_RefAverage, q_file)

with open("../data/test_data/encoded_firstQuestion_RefAverage_testset.pickle", "wb") as q_file:
    pickle.dump(encoded_test_firstQuestion_RefAverage, q_file)

with open("../data/test_data/encoded_prevQuestion_RefAverage_testset.pickle", "wb") as q_file:
    pickle.dump(encoded_test_prevQuestion_RefAverage, q_file)

#get average of first and previous reformulation embeddings
encoded_train_firstQprevQ_RefAverage  = dict()
encoded_dev_firstQprevQ_RefAverage  = dict()
encoded_test_firstQprevQ_RefAverage  = dict()
for qId in encoded_train_questions.keys():
    questions = tf.keras.layers.concatenate([ encoded_train_firstQuestion_RefAverage[qId], encoded_train_prevQuestion_RefAverage[qId]], axis=0)
    avg_questions =  tf.reduce_mean(questions, axis=0)
    avg_questions = tf.expand_dims(avg_questions, 0)
    encoded_train_firstQprevQ_RefAverage[qId] = avg_questions

for qId in encoded_dev_questions.keys():
    questions = tf.keras.layers.concatenate([ encoded_dev_firstQuestion_RefAverage[qId], encoded_dev_prevQuestion_RefAverage[qId]], axis=0)
    avg_questions =  tf.reduce_mean(questions, axis=0)
    avg_questions = tf.expand_dims(avg_questions, 0)
    encoded_dev_firstQprevQ_RefAverage[qId] = avg_questions

for qId in encoded_test_questions.keys():
    questions = tf.keras.layers.concatenate([ encoded_test_firstQuestion_RefAverage[qId], encoded_test_prevQuestion_RefAverage[qId]], axis=0)
    avg_questions =  tf.reduce_mean(questions, axis=0)
    avg_questions = tf.expand_dims(avg_questions, 0)
    encoded_test_firstQprevQ_RefAverage[qId] = avg_questions

with open("../data/train_data/encoded_firstQprevQ_RefAverage_trainset.pickle", "wb") as q_file:
    pickle.dump(encoded_train_firstQprevQ_RefAverage, q_file)

with open("../data/dev_data/encoded_firstQprevQ_RefAverage_devset.pickle", "wb") as q_file:
    pickle.dump(encoded_dev_firstQprevQ_RefAverage, q_file)

with open("../data/test_data/encoded_firstQprevQ_RefAverage_testset.pickle", "wb") as q_file:
    pickle.dump(encoded_test_firstQprevQ_RefAverage, q_file)
