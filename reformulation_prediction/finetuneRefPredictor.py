from transformers import TFBertForSequenceClassification, TFRobertaForSequenceClassification, TFTrainer, TFTrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
import json 
import pickle

"""Fine-tuning BERT for reformulation prediction"""

with open("../data/ref_prediction/dev_encodings", "rb") as dev_file:
   dev_encodings = pickle.load(dev_file)

with open("../data/ref_prediction/dev_labels", "r") as dev_file:
    dev_labels = json.load(dev_file)

with open("../data/ref_prediction/test_encodings", "rb") as train_file:
   test_encodings = pickle.load(train_file)

with open("../data/ref_prediction/test_labels", "r") as train_file:
    test_labels = json.load(train_file)


test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))

#used for training
dev_dataset = tf.data.Dataset.from_tensor_slices((
    dict(dev_encodings),
    dev_labels
))

#calculate precision, recall and f1 scores
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_0_labels': f1_class[0],
        'precision_0_labels': precision_class[0],
        'recall_0_labels': recall_class[0],
        'f1_1_labels': f1_class[1],
        'precision_1_labels': precision_class[1],
        'recall_1_labels': recall_class[1]
    }



training_args = TFTrainingArguments(
    output_dir='../data/ref_prediction/checkpoints/BERT',    # output directory, modify if RoBERTa is used
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    #weight_decay=0.001,               # strength of weight decay
    learning_rate=3e-5,
    logging_dir='../data/ref_prediction/logs_train',   # directory for storing logs
    logging_steps=500,
    save_steps=1000,
    do_train=True

)

with training_args.strategy.scope():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
    #alternatively use roberta:
    #model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)


eval_summary_writer = tf.summary.create_file_writer('../data/ref_prediction/logs_train')


trainer = TFTrainer(
    model=model,                       # the instantiated 🤗 Transformers model to be trained
    args=training_args,                # training arguments, defined above
    train_dataset=dev_dataset,         # training dataset (we fine-tune our model on the data coming from the devset of ConvRef, 
                                       # since predictions will be made during training on ConvRef)
    eval_dataset=test_dataset,         # evaluation dataset
    tb_writer=eval_summary_writer,
    compute_metrics=compute_metrics
)


trainer.train()