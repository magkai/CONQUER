import json
import pickle
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


"""Pre-compute results for reformulation prediction on the ConvRef trainset to enable faster training"""
#only used for TRAINING data not for dev and test data

#load fine-tuned BERT model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
ref_pred_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
ckpt = tf.train.Checkpoint(model=ref_pred_model)
ckpt.restore('../data/ref_prediction/checkpoints/BERT/ckpt-7').expect_partial()

#load json file with all training questions
with open("../data/train_data/all_questions_trainset.json", "r") as qfile:
   questions = json.load(qfile)

with open("../data/train_data/conversations_by_id.json", "r") as conv_file:
   conversations = json.load(conv_file)

with open("../data/train_data/reformulations_by_id.json", "r") as ref_file:
   reformulations = json.load(ref_file)

#question id list
with open("../data/train_data/questionId_list_trainset.json", "r") as conv_file:
   question_list = json.load(conv_file)


#get follow-up intent (nextQuestions)
#get next reformulation for same intent:
   # - real/noisy user setting: use refs as they are in ConvRef (can be empty if no further reformulation is available; last reformulation of an intent)
   # - ideal user setting: get initial question for last reformulation of an intent from ConvRef (ideal scenario where always a reformulation is available)
def getNextQuestionRef():
   nextQuestions = dict()
   nextIdealRefs = dict()
   nextRealRefs = dict()
   for qId in questions.keys():
      splitted = qId.split("-")
      convId = splitted[0]
      q_turn = splitted[1]
      #check if initial question
      if qId.count("-") == 1:
         if len(reformulations[qId]) > 0:
            nextIdealRefs[qId] = reformulations[qId][0]
            nextRealRefs[qId] = reformulations[qId][0]
         #if there is no reformulation take same question again
         else:
            nextIdealRefs[qId] = questions[qId]
      #reformulations:
      else:
         ref_turn = splitted[2]
         main_qid = str(convId) + "-" + str(q_turn)
         #check if another reformulation is available
         if int(ref_turn) < len(reformulations[main_qid])-1:
            nextIdealRefs[qId] = reformulations[main_qid][int(ref_turn)+1]
            nextRealRefs[qId] = reformulations[main_qid][int(ref_turn)+1]
         #no ref available: add initial question in ideal case
         else:
            nextIdealRefs[qId] = questions[main_qid]
      #check if follow-up intent is within the same conversation
      if int(q_turn) < 4:
         nextQuestions[qId] = conversations[str(convId)][int(q_turn)+1]
      else:
         #for the last intent in the final conversation, use the first question of the first conversation
         if int(convId) < 6720:
            nextQuestions[qId] = conversations[str(int(convId)+1)][0]
         else:
            nextQuestions[qId] = conversations["1"][0]
   
   return (nextQuestions, nextIdealRefs, nextRealRefs)


#apply reformulation predictor to determine whether two questions are reformulations from each other
def getRewardFromRefPredictor(question1, question2):
        inputs = bert_tokenizer(question1, question2, padding=True, truncation=True, return_tensors="tf")
        outputs = ref_pred_model(inputs)
        logits = outputs.logits.numpy()
        probs = tf.nn.softmax(logits)
        fraction = tf.reduce_max(probs)   
        preds = logits.argmax(-1)
   
        return (int(preds[0]), float(fraction.numpy()))


nextQuestions, nextIdealRefs, nextRealRefs = getNextQuestionRef()

#store data
with open("../data/train_data/next_questions_trainset.json", "w") as qfile:
   json.dump(nextQuestions,qfile)

with open("../data/train_data/next_reformulations_ideal_trainset.json", "w") as qfile:
   json.dump(nextIdealRefs,qfile)

with open("../data/train_data/next_reformulations_real_trainset.json", "w") as qfile:
   json.dump(nextRealRefs,qfile)


nextInfoPredictions = dict()
nextInfoFracPredictions = dict()
refRealPredictions = dict()
refRealFracPredictions = dict()
refIdealPredictions = dict()
refIdealFracPredictions = dict()

for qId in question_list:
    current_question = questions[qId]
    #get predictions for next info needs, frac_predictions stores the ref prediction probability 
    nextInfoPredictions[qId], nextInfoFracPredictions[qId] = getRewardFromRefPredictor(current_question, nextQuestions[qId])
    #get predictions for reformulations (ideal setting)
    if qId in nextIdealRefs.keys():
        refIdealPredictions[qId], refIdealFracPredictions[qId] = getRewardFromRefPredictor(current_question, nextIdealRefs[qId])   
   #get predictions for reformulations (noisy setting)
    if qId in nextRealRefs.keys():
        refRealPredictions[qId], refRealFracPredictions[qId] = getRewardFromRefPredictor(current_question, nextRealRefs[qId])
    

#store results   
with open("../data/train_data/next_info_predictions_trainset.json", "w") as qfile:
   json.dump(nextInfoPredictions, qfile)

with open("../data/train_data/ref_ideal_predictions_trainset.json", "w") as qfile:
   json.dump(refIdealPredictions, qfile)

with open("../data/train_data/ref_real_predictions_trainset.json", "w") as qfile:
   json.dump(refRealPredictions, qfile)

with open("../data/train_data/next_info_frac_predictions_trainset.json", "w") as qfile:
   json.dump(nextInfoFracPredictions, qfile)

with open("../data/train_data/ref_ideal_frac_predictions_trainset.json", "w") as qfile:
   json.dump(refIdealFracPredictions, qfile)

with open("../data/train_data/ref_real_frac_predictions_trainset.json", "w") as qfile:
   json.dump(refRealFracPredictions, qfile)

print("ref prediction done")