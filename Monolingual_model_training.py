import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import numpy as np

df_train = pd.read_csv("train_hindi_sentiment.csv")
df_val = pd.read_csv("val_hindi_sentiment.csv")
df = pd.read_csv("test_hindi_sentiment.csv")
df['tweet_translated_llama3'] = df['tweet_translated_llama3'].astype(str)
df_train['tweet_translated_llama3'] = df_train['tweet_translated_llama3'].astype(str)
df_val['tweet_translated_llama3'] = df_val['tweet_translated_llama3'].astype(str)
X_train = df_train['tweet_translated_llama3']  
y_train = df_train['sentiment_en']       
X_val = df_val['tweet_translated_llama3']  
y_val = df_val['sentiment_en']
X_test = df['tweet_translated_llama3']    
y_test = df['sentiment_en']

model_name = 'l3cube-pune/hindi-bert-v2'         # Monolingual model for Hindi
#model_name = 'l3cube-pune/tamil-bert'           # Monolingual model for Tamil
#model_name = 'almanach/camembert-base'          # Monolingual model for French
#model_name = 'DeepPavlov/rubert-base-cased'     # Monolingual model for Russian

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)
train_data = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
val_data = Dataset.from_pandas(pd.DataFrame({'text': X_val, 'label': y_val}))
test_data = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))
train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    macro_f1 = f1_score(p.label_ids, preds, average='macro')
    return {'macro_f1': macro_f1}
    training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    eval_steps=50,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1'
    #early_stopping_patience=1
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)
trainer.train()
predictions = trainer.predict(test_data)
predicted_labels = predictions.predictions.argmax(axis=1)
accuracy = accuracy_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels, average='macro')

print(f"SA Accuracy: {accuracy:.3f}")
print(f"SA Macro F1 Score: {f1:.3f}")

df['LlaMa3_trained_mono'] = predicted_labels
#df['GPT_trained_mono'] = predicted_labels         # If monolingual model is being trained on GPT Translations
#df['MT_trained_mono'] = predicted_labels          # If monolingual model is being trained on MT Translations

df.to_csv("Test_hindi_sentiment.csv", index=False)
print("Testing complete and saved for LlaMa-3 Hindi Sentiment")