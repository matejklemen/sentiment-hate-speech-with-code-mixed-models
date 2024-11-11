#----------------------------------------------------------------------SENTIMENT ANALYSIS-----------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
import trl
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("Hindi_sentiment.csv") # adapted to the different datasets we use
test_df = pd.read_csv("Test_hindi_sentiment.csv") # adapted to the different datasets we use
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[['tweet', 'sentiment']]
X_eval = val_df[['tweet', 'sentiment']]
X_test = test_df[['tweet', 'sentiment']]
def generate_prompt(data_point):
    return f"""
            Classify the text into Positive, Negative, Neutral, and return the answer as the corresponding sentiment.
text: {data_point["tweet"]}
label: {data_point["sentiment"]}""".strip()

def generate_test_prompt(data_point):
    return f"""
            Classify the text into Positive, Negative, Neutral, and return the answer as the corresponding sentiment.
text: {data_point["tweet"]}
label: """.strip()
X_train.loc[:,'text'] = X_train.apply(generate_prompt, axis=1)
X_eval.loc[:,'text'] = X_eval.apply(generate_prompt, axis=1)
y_true = X_test.loc[:,'sentiment']
X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])
X_train.sentiment.value_counts()
train_data = Dataset.from_pandas(X_train[["text"]])
eval_data = Dataset.from_pandas(X_eval[["text"]])
base_model_name = "meta-llama/Llama-3.1-8B"
access_token = ""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config,
    token=access_token
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
tokenizer.pad_token_id = tokenizer.eos_token_id
def predict(test, model, tokenizer):
    y_pred = []
    categories = ["Positive", "Negative", "Neutral"]    
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=2, 
                        temperature=0.1)        
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")  
    return y_pred
y_pred = predict(X_test, model, tokenizer)
def evaluate(y_true, y_pred):
    labels = ["Positive", "Negative", "Neutral"]
    mapping = {label: idx for idx, label in enumerate(labels)}   
    def map_func(x):
        return mapping.get(x, -1)
    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')
    unique_labels = set(y_true_mapped)  # Get unique labels
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Macro F1 Score HINDI SA Non-CM: {macro_f1:.3f}')
evaluate(y_true, y_pred)
import bitsandbytes as bnb
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    #return list(lora_module_names)
    return [mod for mod in lora_module_names if 'down_proj' not in mod]
modules = find_all_linear_names(model)
modules
output_dir="Llama3.1-8B_hindi_sentiment"
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=1,                         
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps = 0.2
)
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False,
    dataset_kwargs={
    "add_special_tokens": False,
    "append_concat_token": False,
    }
)
trainer.train()
model.config.use_cache = True
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
y_pred = predict(X_test, model, tokenizer)
evaluate(y_true, y_pred)
test_df["predicted_sentiment_llama3.1-8B"] = y_pred
test_df.to_csv("test_predictions_Llama3.1-8B_Hindi_sentiment.csv", index=False)
print(f"Test predictions saved to test_predictions_Llama3.1-8B_Hindi_sentiment.csv")


#------------------------------------------------------------------OFFENSIVE SPEECH PREDICTION---------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
import trl
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("Hindi_sentiment.csv") # adapted to the different datasets we use
test_df = pd.read_csv("Test_hindi_sentiment.csv") # adapted to the different datasets we use
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[['tweet', 'label_en']]
X_eval = val_df[['tweet', 'label_en']]
X_test = test_df[['tweet', 'label_en']]
def generate_prompt(data_point):
    return f"""
            Classify the text into 0 (Not Toxic), 1 (Toxic), and return the answer as the corresponding toxicity label.
text: {data_point["tweet"]}
label: {data_point["label_en"]}""".strip()
def generate_test_prompt(data_point):
    return f"""
            Classify the text into 0 (Not Toxic), 1 (Toxic), and return the answer as the corresponding toxicity label.
text: {data_point["tweet"]}
label: """.strip()
X_train.loc[:,'text'] = X_train.apply(generate_prompt, axis=1)
X_eval.loc[:,'text'] = X_eval.apply(generate_prompt, axis=1)
y_true = test_df.loc[:,'label_en']
X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])
X_train.label_en.value_counts()
train_data = Dataset.from_pandas(X_train[["text"]])
eval_data = Dataset.from_pandas(X_eval[["text"]])
base_model_name = "meta-llama/Llama-3.1-8B"
access_token = ""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config,
    token=access_token
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
tokenizer.pad_token_id = tokenizer.eos_token_id
def predict(test, model, tokenizer):
    y_pred = []
    categories = ['1','0']   
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=2, 
                        temperature=0.1)       
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(int(category))
                break
        else:
            y_pred.append(-1)   
    return y_pred
y_pred = predict(X_test, model, tokenizer)
def evaluate(y_true, y_pred):
    labels = [0, 1]
    target_names = ["Not Toxic", "Toxic"]
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    unique_labels = set(y_true)
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {target_names[label]}: {label_accuracy:.3f}')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Macro F1 Score HINDI HS Non-CM: {macro_f1:.3f}')
evaluate(y_true, y_pred)
import bitsandbytes as bnb
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    #return list(lora_module_names)
    return [mod for mod in lora_module_names if 'down_proj' not in mod]
modules = find_all_linear_names(model)
modules
output_dir="Llama3.1-8B_hindi_of"
peft_config = LoraConfig(
    lora_alpha=4,
    lora_dropout=0,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=1,                         
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps = 0.2
)
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=256,
    packing=False,
    dataset_kwargs={
    "add_special_tokens": False,
    "append_concat_token": False,
    }
)
trainer.train()
model.config.use_cache = True
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
y_pred = predict(X_test, model, tokenizer)
evaluate(y_true, y_pred)
test_df["predicted_label_llama3.1-8B"] = y_pred
test_df.to_csv("test_predictions_Llama3_8B_Hindi_toxic.csv", index=False)
print(f"Test predictions saved as test_predictions_Llama3_8B_Hindi_toxic.csv")
