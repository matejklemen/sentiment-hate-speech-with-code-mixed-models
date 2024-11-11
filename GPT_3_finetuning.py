#################################################################### SENTIMENT ANALYSIS ############################################################################

import os
from pathlib import Path
!pip install datasets
!pip install Openai
from datasets import load_dataset
#from dotenv import load_dotenv
#load_dotenv()
from openai import OpenAI
import pandas as pd
import csv
import json

df = pd.read_csv(csv_file_path)
df.rename(columns = {'tweet':'prompt','sentiment_en':'completion'}, inplace = True)
df['prompt'] = df['prompt'].str.strip()
#df['completion'] = df['completion'].str.strip()
df['prompt'] = df['prompt'] + "\n\nSentiment:\n\n"
#df['completion'] = " "+df['completion'] + " END"
df.to_json("gpt_train_prepared.jsonl", orient='records', lines=True)

!openai tools fine_tunes.prepare_data -f gpt_train.jsonl
train_path = "/content/gpt_train_prepared.jsonl"
train_id = "file-6PDxzsGO1NIXUrg5q3syxM8N"
fine_tune_id = "ftjob-76DoNZyAO45zQaIUq7mqFYAz"
DO_UPLOAD = False
DO_FINETUNE = False
DO_CHECK =  True


data = load_dataset("json", data_files={
    "train": train_path,
})

api_key =""
client = OpenAI(
    api_key=api_key
)

if DO_UPLOAD:
    print("Uploading data...")
    upload_response = client.files.create(
        file=Path(train_path),
        purpose="fine-tune"
    )

    print(upload_response)
    print(upload_response.id)
    exit(0)

if DO_FINETUNE:
    fine_tune_response = client.fine_tuning.jobs.create(
        model="davinci-002",  # Firstly used curie, which is now deprecated, davinci-002 is the suggested replacement
        training_file=train_id,
        hyperparameters={"n_epochs": 2},
        #classification_n_classes=3
    )

    print(fine_tune_response)

if DO_CHECK:
    if fine_tune_id is None:
        print("fine_tune_id is set to None (no-op)")
        exit(0)

    retrieve_response = client.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tune_id)
    print(retrieve_response)
client.fine_tuning.jobs.list_events(fine_tune_id)

# USING THE MODEL
fine_tuned_model_id = 'ftjob-76DoNZyAO45zQaIUq7mqFYAz'
api_key =""
retrieve_response = client.fine_tuning.jobs.retrieve(fine_tuned_model_id)
fine_tuned_model = retrieve_response.fine_tuned_model
#!pip install tiktoken
import tiktoken

tiktoken.encoding_name_for_model('davinci-002')

tokenizer = tiktoken.encoding_for_model('davinci-002')
tokens = ["0", "1", "2"]
token_id = tokenizer.encode_single_token("0")
token_id

test_data = pd.read_csv(test_data_path)
test_data.drop('Unnamed: 0', axis = 'columns', inplace = True)
test_data.drop('sentiment', axis = 'columns', inplace = True)
test_data.columns = ['prompt','completion']
test_data['prompt'] = test_data['prompt'].str.strip()
#test_data['completion'] = test_data['completion'].str.strip()
test_data['prompt'] = test_data['prompt'] + "\n\nSentiment:\n\n"
#test_data['completion'] = " "+test_data['completion'] + " end"

predictions = []

logit_bias = {
    15: 50,
    16: 50,
    17: 50
}
for tweet in test_data["prompt"]:

    response = client.completions.create(
        model=fine_tuned_model,
        prompt=tweet,
        max_tokens=1,
        temperature = 0.1,
        logit_bias=logit_bias
        #stop=[" end"]
    )
    #predictions.append(response.choices[0].text.strip())
    predictions.append(response.choices[0].text.strip())

test_data["predicted_sentiment"] = predictions

tweet = "Kaha salman khan\n\nSentiment:\n\n"

response = client.completions.create(
    model=fine_tuned_model,
    prompt=tweet,
    max_tokens = 2,
    #stop=[" end"]
)
#print(response['choices'][0]['prompt'])
response.choices[0]

#################################################################### OFFENSIVE SPEECH PREDICTION ##################################################################


import os
from pathlib import Path
!pip install datasets
!pip install Openai
from datasets import load_dataset
#from dotenv import load_dotenv
#load_dotenv()
from openai import OpenAI
import pandas as pd
import csv
import json

df = pd.read_csv(csv_file_path)
df.rename(columns = {'tweet':'prompt','label_en':'completion'}, inplace = True)
df['prompt'] = df['prompt'].str.strip()
#df['completion'] = df['completion'].str.strip()
df['prompt'] = df['prompt'] + "\n\nLabel:\n\n"
#df['completion'] = " "+df['completion'] + " END"
df.to_json("gpt_train_prepared.jsonl", orient='records', lines=True)

!openai tools fine_tunes.prepare_data -f gpt_train.jsonl
train_path = "/content/gpt_train_prepared.jsonl"
train_id = "file-6PDxzsGO1NIXUrg5q3syxM8N" #changes every time the code is run 
fine_tune_id = "ftjob-76DoNZyAO45zQaIUq7mqFYAz"
DO_UPLOAD = False
DO_FINETUNE = False
DO_CHECK =  True


data = load_dataset("json", data_files={
    "train": train_path,
})

api_key =""
client = OpenAI(
    api_key=api_key
)

if DO_UPLOAD:
    print("Uploading data...")
    upload_response = client.files.create(
        file=Path(train_path),
        purpose="fine-tune"
    )

    print(upload_response)
    print(upload_response.id)
    exit(0)

if DO_FINETUNE:
    fine_tune_response = client.fine_tuning.jobs.create(
        model="davinci-002",  # Firstly used curie, which is now deprecated, davinci-002 is the suggested replacement
        training_file=train_id,
        hyperparameters={"n_epochs": 2},
        #classification_n_classes=3
    )

    print(fine_tune_response)

if DO_CHECK:
    if fine_tune_id is None:
        print("fine_tune_id is set to None (no-op)")
        exit(0)

    retrieve_response = client.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tune_id)
    print(retrieve_response)
client.fine_tuning.jobs.list_events(fine_tune_id)

# USING THE MODEL
fine_tuned_model_id = 'ftjob-76DoNZyAO45zQaIUq7mqFYAz'
api_key =""
retrieve_response = client.fine_tuning.jobs.retrieve(fine_tuned_model_id)
fine_tuned_model = retrieve_response.fine_tuned_model
#!pip install tiktoken
import tiktoken
tiktoken.encoding_name_for_model('davinci-002')
tokenizer = tiktoken.encoding_for_model('davinci-002')
tokens = ["0", "1"]
token_id = tokenizer.encode_single_token("0")
token_id

test_data = pd.read_csv(test_data_path)
test_data.drop('Unnamed: 0', axis = 'columns', inplace = True)
test_data.drop('label', axis = 'columns', inplace = True)
test_data.columns = ['prompt','completion']
test_data['prompt'] = test_data['prompt'].str.strip()
#test_data['completion'] = test_data['completion'].str.strip()
test_data['prompt'] = test_data['prompt'] + "\n\nLabel:\n\n"
#test_data['completion'] = " "+test_data['completion'] + " end"

predictions = []
logit_bias = {
    15: 50,
    16: 50,
}
for tweet in test_data["prompt"]:

    response = client.completions.create(
        model=fine_tuned_model,
        prompt=tweet,
        max_tokens=1,
        temperature = 0.1,
        logit_bias=logit_bias
        #stop=[" end"]
    )
    #predictions.append(response.choices[0].text.strip())
    predictions.append(response.choices[0].text.strip())

test_data["predicted_label"] = predictions
