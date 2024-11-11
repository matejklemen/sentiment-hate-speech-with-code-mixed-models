import json
import openai
import pandas as pd
api_key =""
openai.api_key = api_key

df = pd.read_csv('conda/envs/gpu_env/Cross_lingual/datasets/Final_datasets/Hindi/train.csv')
# df = pd.read_csv('conda/envs/gpu_env/Cross_lingual/datasets/Final_datasets/French/train.csv')
# df = pd.read_csv('conda/envs/gpu_env/Cross_lingual/datasets/Final_datasets/Tamil/train.csv')
# df = pd.read_csv('conda/envs/gpu_env/Cross_lingual/datasets/Final_datasets/Russian/train.csv')

def prompt_creation(sample):
    INPUT_KEY = "Input"
    tokens = sample.split()
    max_tokens = 512
    truncated_tokens = tokens[:max_tokens]
    truncated_input = " ".join(truncated_tokens)
    input_text = f"{INPUT_KEY} : {str(truncated_input)} "
    return input_text
    
def completion(sample):
    RESPONSE_KEY = ""
    response = f"{RESPONSE_KEY}Label : {sample}"
    return response
    
# df['prompt']= df.apply(prompt_creation,axis=1)
df['prompt'] = df['tweet'].apply(prompt_creation)
df['completion'] = df['sentiment'].apply(completion)
# df['prompt'] = df.apply(prompt_creation,axis=1)

sub_df = df[['prompt','completion']].copy()
dataset_1 = sub_df.to_dict(orient='records')
# save the json file 
file_name = 'gpt_ft_.jsonl'
with open(file_name, 'w') as f:
    for record in dataset_1:
        f.write(json.dumps(record) + '\n')
        
!pip install openai
!openai tools fine_tunes.prepare_data -f gpt_ft.jsonl

from openai import OpenAI
# client = OpenAI()
client = OpenAI(api_key)

from openai import OpenAI
# client = OpenAI()

upload_response = client.files.create(
  file=open("gpt_ft.jsonl", "rb"),
  purpose="fine-tune"
)
file_id = upload_response.id
file_id

response = client.fine_tuning.jobs.create(
  training_file=file_id, 
  # model="curie-001"   # this variant was depritiated later and replaced with davinci for continuing the research tests
  model="davinci-002",
  hyperparameters={
    "n_epochs":2,
    "learning_rate_multiplier":0.1,
  }
)
response

job_id = response.id
status = response.status

print(f'Fine-tunning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")

status = response.status
print(status)

fine_tune_events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
fine_tune_events

retrieve_response = client.fine_tuning.jobs.retrieve(response.id)
retrieve_response

if retrieve_response.fine_tuned_model != None :
    fine_tuned_model = response.fine_tuned_model
    print(fine_tuned_model)
else :
    print("running")

# Option 2 | if response.fine_tuned_model == null
retrieve_response = client.fine_tuning.jobs.retrieve(response.id)
fine_tuned_model = retrieve_response.fine_tuned_model
fine_tuned_model

def generated_output(prompt):
    new_prompt = f'Input:{prompt}'
    answer = client.completions.create(
      model=fine_tuned_model,
      prompt=new_prompt,
      # max_tokens=5,
      # temperature=0.1
    )
    return answer.choices[0].text

test = pd.read_csv('conda/envs/gpu_env/Cross_lingual/datasets/Final_datasets/Hindi/test.csv')
test['output'] = test['tweet'].apply(generated_output)
import re
def pred_sentiment(sample):
    if re.search(r'\bpositive\b', sample, flags=re.IGNORECASE):
        label =  'Positive'
    elif re.search(r'\bNegative\b', sample, flags=re.IGNORECASE):
        label = 'Negative'
    elif re.search(r'\bneutral\b', sample, flags=re.IGNORECASE):
        label = 'Neutral'
    elif re.search(r'\boffensive\b', sample, flags=re.IGNORECASE):
        label = 'offensive'
    else:
        label = 'Unknown'
    return label
    
import swifter
test['pred'] = test['output'].swifter.apply(pred_sentiment)
from sklearn.metrics import accuracy_score,f1_score
predicted = df['pred']
actual = df['output']
accuracy = accuracy_score(actual, predicted)
fscore = f1_score(actual, predicted, average='weighted')
print(f"F-score: {fscore}")
print(f"Accuracy: {accuracy}")