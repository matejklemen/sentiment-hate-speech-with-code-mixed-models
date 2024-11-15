# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as plt
import swifter
import os
import torch
from transformers import AutoTokenizer, AutoModel, BertModel, AutoModelForMaskedLM,BertTokenizer,TFRobertaModel
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score
import torch
import torch.nn.functional as F
import tensorflow as tf
##########################################################################
'''
The piece of code loads train,val and test dataset
We label encode the dataset also
Uncomment the tokenizer corresponding to model
Code to preprocess text data - also present
'''
##########################################################################

# '''Hindi Dataset'''
# train = pd.read_csv('datasets/Final_datasets/Hindi/train.csv')
# # test = pd.read_csv('datasets/Final_datasets/Hindi/test.csv')
# test = pd.read_csv('datasets/Final_datasets/Hindi_SA_test50.csv')
# val = pd.read_csv('datasets/Final_datasets/Hindi/val.csv')

# '''Hindi of Dataset'''
train = pd.read_csv('datasets/Final_datasets/hindi_of/train.csv')
# test = pd.read_csv('datasets/Final_datasets/hindi_of/test.csv')
test = pd.read_csv('datasets/Final_datasets/Hindi_HS_test100.csv')
val = pd.read_csv('datasets/Final_datasets/hindi_of/val.csv')

# '''French Dataset'''
# train = pd.read_csv('datasets/Final_datasets/French/train.csv')
# # test = pd.read_csv('datasets/Final_datasets/French/test.csv')
# test = pd.read_csv('datasets/Final_datasets/French_SA_Test50.csv')
# val = pd.read_csv('datasets/Final_datasets/French/val.csv')

# ''' FRENCH of Dataset'''
# train = pd.read_csv('datasets/Final_datasets/French_of/train.csv')
# # # test = pd.read_csv('datasets/Final_datasets/French_of/test.csv')
# test = pd.read_csv('datasets/Final_datasets/French_HS_TestVal50.csv')
# val = pd.read_csv('datasets/Final_datasets/French_of/val.csv')

# # # '''Russian Dataset'''
# train = pd.read_csv('datasets/Final_datasets/Russian/train.csv')
# # test = pd.read_csv('datasets/Final_datasets/Russian/test.csv')
# val = pd.read_csv('datasets/Final_datasets/Russian/val.csv')

# '''Russian of '''
# train = pd.read_csv('datasets/Final_datasets/Russian_of/train.csv')
# # # test = pd.read_csv('datasets/Final_datasets/Russian_of/test.csv')
# test = pd.read_csv('datasets/Final_datasets/Russian_HS_test50.csv')
# val = pd.read_csv('datasets/Final_datasets/Russian_of/val.csv')

# '''Tamil Dataset'''
# train = pd.read_csv('datasets/Final_datasets/Tamil/train.csv')
# # test = pd.read_csv('datasets/Final_datasets/Tamil/test.csv')
# test = pd.read_csv('datasets/Final_datasets/Tamil_landetect_50.csv')
# val = pd.read_csv('datasets/Final_datasets/Tamil/val.csv')

# '''Tamil offensive Dataset'''
# train = pd.read_csv('datasets/Final_datasets/Tamil_of/train.csv')
# # test = pd.read_csv('datasets/Final_datasets/Tamil_of/test.csv')
# test = pd.read_csv('datasets/Final_datasets/Tamil_HS_langdetect_50.csv')
# val = pd.read_csv('datasets/Final_datasets/Tamil_of/val.csv')

X_train = train['tweet']
X_val = val['tweet']
X_test = test['tweet']
# y_train = train['sentiment_en']
# y_val = val['sentiment_en']
# y_test = test['sentiment_en']
y_train = train['label_en']
y_val = val['label_en']
y_test = test['label_en']

out_dir = './models/hindi_hs_models/mbert'

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name' if present
    text = re.sub(r'(@.*?)[\s]', ' ', text) if re.search(r'@.*?[\s]', text) else text

    # Replace '&amp;' with '&' if present
    text = re.sub(r'&amp;', '&', text) if '&amp;' in text else text

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text



##########################################################################
'''
The piece of code loads the data model and tokenizer
'''
##########################################################################

# print('roberta tamil')
# DeepPavlov/rubert-base-cased
# tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

# tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
# model = AutoModel.from_pretrained("google/muril-base-cased")

# tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/tamil-bert")
# model = AutoModel.from_pretrained("l3cube-pune/tamil-bert")

## '''HingRoberta'''
# tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hing-roberta")
# model = AutoModel.from_pretrained("l3cube-pune/hing-roberta")

# Load model directly
# tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base")
# # model = AutoModel.from_pretrained("Twitter/twhin-bert-base")

# # '''mBert'''
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# '''Roberta'''
# model_name = "xlm-roberta-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

def preprocess_text(text):
    inputs = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        truncation=True,
        # return_tensors="pt",
        return_attention_mask=True
    )
    return inputs

def processing_data(data):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    for sent in data:

      # Recieve the inputs
      inputs = preprocess_text(sent)

      # Add the outputs to the lists
      input_ids.append(inputs.get('input_ids'))
      attention_masks.append(inputs.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

# Print sentence 0 and its encoded token ids
# Example of preparing input for BERT
# input_ids = torch.LongTensor([input_ids])
# token_ids = list(processing_data([X_train[0]])[0].squeeze().numpy())
# print('Original: ', X_train[0])
# print('Token IDs: ', token_ids)

# Tokenizing the train and the validation data
train_inputs, train_masks = processing_data(X_train)
val_inputs, val_masks = processing_data(X_val)
test_inputs,test_masks = processing_data(X_test)

""" PyTorch DataLoader"""

#create an iterator for our dataset using the torch DataLoader class. This will help save on memory during training and boost the training speed
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train.tolist())
val_labels = torch.tensor(y_val.tolist())
test_labels = torch.tensor(y_test.tolist())

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
batch_size = 16

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Specify GPU device(s)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



# TRAIN
# # BertForSequenceClassification Model
# from transformers import BertForSequenceClassification
# from transformers import AdamW, get_linear_schedule_with_warmup

# # Load the pre-trained model
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels= 2 )

# # Set the model to training mode
# model.train()

import torch.nn as nn
from transformers import BertModel

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        # D_in, H, D_out = 768, 50, 3
        D_in, H, D_out = 768, 100, 3

        # Instantiate BERT model
        self.bert = model

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits



from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification


def initialize_model(epochs=2, num_labels=3):
    """Initialize the Bert Classifier, the optimizer, and the learning rate scheduler for sequence classification.

    Args:
        epochs (int): Number of training epochs.
        num_labels (int): Number of output labels in the classification task.
        device (str): Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        bert_classifier: Initialized BERT sequence classification model.
        optimizer: AdamW optimizer for model parameter updates.
        scheduler: Learning rate scheduler.
    """
    # # Instantiate BertForSequenceClassification
    # bert_classifier = BertForSequenceClassification.from_pretrained(
    #     model_name,  # Choose the appropriate BERT variant
    #     num_labels= 2,  # Set the number of output labels
    #     output_attentions=False,
    #     output_hidden_states=False,
    # )

    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)
    # Move the model to the specified device
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(
        bert_classifier.parameters(),
        lr=5e-5,    # Default learning rate
        eps=1e-8    # Default epsilon value
    )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value
        num_training_steps=total_steps
    )

    return bert_classifier, optimizer, scheduler

import random
import time
import torch.nn as nn

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
  """ Set seed for reproducibility. """
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  # torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")

    print("Training complete!")

def save_model(model, tokenizer, output_dir= out_dir):
    # Create output directory if it doesn't exist 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model state dictionary
    model_path = os.path.join(output_dir, 'bert_classifier_model.pth')
    torch.save(model.state_dict(), model_path)

    # Save the tokenizer
    tokenizer_path = os.path.join(output_dir, 'bert_tokenizer')
    tokenizer.save_pretrained(tokenizer_path)

    print(f"Model and tokenizer saved at: {output_dir}")



def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking for better performance
torch.cuda.empty_cache()  # Clear GPU cache to release memory

set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
train(bert_classifier, train_dataloader, val_dataloader, epochs=5, evaluation=True)
# Save the trained model and tokenizer
save_model(bert_classifier, tokenizer)

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

y_scores = bert_predict(bert_classifier, test_dataloader)

# Initialize the output labels
labels = []
# Assuming you have the ground truth labels for the test set in 'true_labels'
# and the probability scores in 'probs'

# Assign predicted class labels
predicted_labels = torch.argmax(torch.tensor(y_scores), dim=1).numpy()
true_labels = y_test.tolist()
# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate F-score
f_score = f1_score(true_labels, predicted_labels, average='weighted')

print("Accuracy:", accuracy)
print("F-score:", f_score)
