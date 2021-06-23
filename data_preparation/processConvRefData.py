import json
import sys
sys.path.append("../utils")
import utils as ut

"""Process and store data from the ConvRef benchmark for later usage"""

#load data
with open("../data/ConvRef/ConvRef_trainset.json") as json_file:
    train_data = json.load(json_file)

with open("../data/ConvRef/ConvRef_devset.json") as json_file:
    dev_data = json.load(json_file)

with open("../data/ConvRef/ConvRef_testset.json") as json_file:
    test_data = json.load(json_file)


def processDataset(data):
    #question id: question
    all_questions = dict()
    #question id: list with processed gold answers
    all_answers = dict()
    #conv id: list with initial questions
    conversations = dict()
    #question id: list with reformulations
    reformulations = dict()
    #list with all questions/reformulation ids
    qList = []
    for conv in data:
        conv_id = conv["conv_id"]
        conversations[conv_id] = []
        for question_info in conv['questions']:
            question_id = question_info['question_id']
            qList.append(question_id)
            question = question_info['question']
            all_questions[question_id] = question
            conversations[conv_id].append(question)
            all_answers[question_id] = ut.getGoldAnswers(question_info["gold_answer"])
            reformulations[question_id] = []
            for ref in question_info["reformulations"]:
                ref_id = ref["ref_id"]
                qList.append(ref_id)
                all_questions[ref_id] = ref["reformulation"]
                reformulations[question_id].append(ref["reformulation"])
                all_answers[ref_id] = ut.getGoldAnswers(question_info["gold_answer"])

    return all_questions, all_answers, qList, conversations, reformulations, 

train_questions, train_answers, train_qList, train_conversations, train_reformulations = processDataset(train_data)
dev_questions, dev_answers, dev_qList, _, _ = processDataset(dev_data)
test_questions, test_answers, test_qList, _, _ = processDataset(test_data)
         
#store train data
with open("../data/train_data/all_questions_trainset.json", "w") as question_file:
    json.dump(train_questions, question_file)

with open("../data/train_data/all_answers_trainset.json", "w") as answer_file:
    json.dump(train_answers, answer_file)

with open("../data/train_data/conversations_by_id.json", "w") as conv_file:
    json.dump(train_conversations, conv_file)

with open("../data/train_data/reformulations_by_id.json", "w") as ref_file:
    json.dump(train_reformulations,ref_file)

with open("../data/train_data/questionId_list_trainset.json", "w") as conv_file:
    json.dump(train_qList, conv_file)

#store dev data
with open("../data/dev_data/all_questions_devset.json", "w") as question_file:
    json.dump(dev_questions, question_file)

with open("../data/dev_data/all_answers_devset.json", "w") as answer_file:
    json.dump(dev_answers, answer_file)

with open("../data/dev_data/questionId_list_devset.json", "w") as conv_file:
    json.dump(dev_qList, conv_file)

#store test data
with open("../data/test_data/all_questions_testset.json", "w") as question_file:
    json.dump(test_questions, question_file)

with open("../data/test_data/all_answers_testset.json", "w") as answer_file:
    json.dump(test_answers, answer_file)

with open("../data/test_data/questionId_list_testset.json", "w") as conv_file:
    json.dump(test_qList, conv_file)


