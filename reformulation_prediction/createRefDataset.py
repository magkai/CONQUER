import tensorflow as tf
import json
import pickle
from transformers import BertTokenizer
import random

"""Create the training data for fine-tuning BERT"""

#ConvRef devset is used for creating training dataset for reformulation prediction
#evaluation done on the ConvRef testset
with open("../data/ConvRef/ConvRef_devset.json") as json_file:
    dev_data = json.load(json_file)

with open("../data/ConvRef/ConvRef_testset.json") as json_file:
    test_data = json.load(json_file)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#create data:
#question/reformulation are paired as positve examples
#question/other question/reformulation from same conversation are used as negative examples
def create_Pairs(data):
    refPairs = []
    pos = 0
    neg = 0
    for conv in data:    
        j = 0  
        for question_info in conv['questions']:  
            question = question_info["question"]
            for k in range(j+1, 5):
                other_question = conv["questions"][k]["question"]
                refPairs.append({"q1": question, "q2":other_question, "label": 0})
                neg +=1
                for i in range(len(conv["questions"][k]["reformulations"])):
                    other_reformulation = conv["questions"][k]["reformulations"][i]["reformulation"]
                    refPairs.append({"q1": question, "q2":other_reformulation, "label": 0})
                    neg +=1


            for i in range(len(question_info["reformulations"])):
                reformulation = question_info["reformulations"][i]["reformulation"]
                refPairs.append({"q1": question, "q2":reformulation, "label": 1})
                pos +=1
                for l in range(i+1, len(question_info["reformulations"])):
                    next_reformulation = question_info["reformulations"][l]["reformulation"]
                    refPairs.append({"q1": reformulation, "q2":next_reformulation, "label": 1})
                    pos +=1
                for k in range(j+1, 5):
                    for n in range(len(conv["questions"][k]["reformulations"])):
                        other_reformulation = conv["questions"][k]["reformulations"][n]["reformulation"]
                        refPairs.append({"q1": reformulation, "q2":other_reformulation, "label": 0})
                        neg +=1
              
            j += 1
    
    return refPairs


def getLabelList(refPairs):
    labels = []
    for pair in refPairs:
        labels.append(pair["label"]) 
    return labels

def getFirstQuestion(refPairs):
    first = []
    for pair in refPairs:
        first.append(pair["q1"]) 
    return first

def getSecondQuestion(refPairs):
    second = []
    for pair in refPairs:
        second.append(pair["q2"]) 
    return second

def processData(data):
    #create reformulation pairs
    refPairs = create_Pairs(data)
    random.seed(4)
    random.shuffle(refPairs)
    #get first/second question in pair
    firstQuestions = getFirstQuestion(refPairs)
    secondQuestions = getSecondQuestion(refPairs)
    #get tokenized input to BERT
    encodings = tokenizer(firstQuestions, secondQuestions, padding=True, truncation=True, return_tensors="tf")
    labels = getLabelList(refPairs)
    return encodings, labels, firstQuestions, secondQuestions

dev_encodings, dev_labels, dev_firstQuestions, dev_secondQuestions = processData(dev_data)
test_encodings, test_labels, test_firstQuestions, test_secondQuestions = processData(test_data)

with open("../data/ref_prediction/dev_firstQuestions", "w") as dev_file:
   json.dump(dev_firstQuestions, dev_file)

with open("../data/ref_prediction/dev_secondQuestions", "w") as dev_file:
   json.dump(dev_secondQuestions, dev_file)

with open("../data/ref_prediction/dev_labels", "w") as dev_file:
    json.dump(dev_labels, dev_file)

with open("../data/ref_prediction/dev_encodings", "wb") as dev_file:
    pickle.dump(dev_encodings, dev_file)

with open("../data/ref_prediction/test_firstQuestions", "w") as dev_file:
   json.dump(test_firstQuestions, dev_file)

with open("../data/ref_prediction/test_secondQuestions", "w") as dev_file:
   json.dump(test_secondQuestions, dev_file)

with open("../data/ref_prediction/test_labels", "w") as dev_file:
    json.dump(test_labels, dev_file)

with open("../data/ref_prediction/test_encodings", "wb") as dev_file:
    pickle.dump(test_encodings, dev_file)

print("Successfully created datasets for reformulation prediction")