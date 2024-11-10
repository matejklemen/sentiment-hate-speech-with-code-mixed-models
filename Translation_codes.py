#---------------------------------------------------------------------GPT TRANSLATION-------------------------------------------------------------------------

import openai
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train_hindi_offensive.csv', encoding='utf-8') # adapted to the different datasets we use
df.reset_index(drop=True, inplace=True)
label_encoder = LabelEncoder()
df['tweet'] = df['tweet'].astype(str)
openai.api_key = ""
texts = df['tweet']
def translate_code_mixed(text):  # we change the prompt to adapt it to all the language sets we consider in our research
    prompt = f"Translate the following hindi-english (hinglish) code-mixed sentence to Hindi. Ensure the translation is valid and in proper Hindi Unicode characters:\n\n{text}\n\nTranslation:"
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=256,
        )
        translation = response.choices[0].text.strip()
        return translation
    except Exception as e:
        print(f"Error translating: {e}")
        return text
df['translated_tweet_GPT'] = df['tweet'].apply(translate_code_mixed)
df.to_csv("train_hindi_offensive.csv", index=False)
print("GPT Translation complete and saved for Hindi offensive .")

#--------------------------------------------------------------------LLAMA-3.1-8B TRANSLATION-------------------------------------------------------------------

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm 

df = pd.read_csv("train_hindi_sentiment.csv") # adapted to the different datasets we use
output_file = 'train_hindi_sentiment.csv'
source = "code mixed hindi"
output_dir = 'lamma_trans_3_hindi'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
access_token = ""
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=access_token
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    use_auth_token=access_token
)
model.config.use_cache = False  # Disable cache if needed
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

def translate_text(input_text, source_language="code mixed hinglish (hindi-english)", target_language="hindi"): # we change the prompt to adapt it to all the language sets we consider in our research
    prompt = (
        f"Translate the following text from {source_language} to {target_language}.\n"
        f"Ensure the translation is valid and in proper hindi Unicode characters/subscript. \n"
        f"Only provide the translation without any additional text.\n\n"
        f"Text: {input_text}\n\n"
        f"Translation:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = translated_text.split("Translation:")[-1].strip()
    return translation

def translate_text_with_null(input_text):
    try:
        return translate_text(input_text)
    except Exception as e:
        print(f"Error translating tweet: {e}")
        return ""  
batch_size = 100
start_batch = 1
start_row = 0
if __name__ == "__main__":
    total_rows = len(df)
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)
    start_idx = (start_batch - 1) * batch_size + start_row
    tqdm.pandas(desc="Translating tweets")
    for batch_num in range(start_batch, num_batches + 1):
        batch_start_idx = (batch_num - 1) * batch_size + start_row
        batch_end_idx = min(batch_start_idx + batch_size, total_rows)
        print(f"Processing batch {batch_num}/{num_batches} from row {batch_start_idx} to {batch_end_idx}...")
        print("HINDI TRAIN SENTIMENT")
        df_batch = df.iloc[batch_start_idx:batch_end_idx]
        df_batch['tweet_translated_llama3'] = df_batch['tweet'].progress_apply(translate_text_with_null)
        output_file = f'output_llama_hindi_sentiment_batch_{batch_num}.csv'
        df_batch.to_csv(output_file, mode='w', index=False)
        start_row = 0

#-----------------------------------------------------------------------MT TRANSLATION--------------------------------------------------------------------------

import pandas as pd
from googletrans import Translator

df = pd.read_csv('train_hindi_sentiment.csv') # adapted to the different datasets we use
df.reset_index(drop=True, inplace=True)
df['tweet'] = df['tweet'].astype(str)
def translate_tweets(df, column_name, dest_lang='fr'):
    translator = Translator()
    def translate_text(text):
        try:
            detected_lang = translator.detect(text).lang
            translated_text = translator.translate(text, src=detected_lang, dest=dest_lang)
            return translated_text.text
        except Exception as e:
            return text
    df[f"{column_name}_translated_MT"] = df[column_name].apply(translate_text)
    return df
df_translated = translate_tweets(df, 'tweet')
df = df_translated
df.to_csv("train_hindi_sentiment.csv", index=False)
print("Translation complete. Translated dataset saved TRAIN HINDI SENTIMENT")