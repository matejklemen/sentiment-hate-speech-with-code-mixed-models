# Research Project: Code and Resources

This repository contains the code and resources for our research project, which explores fine-tuning large language models (LLMs), training monolingual bilingual fewlingual and massively-multilingual models, and translating datasets for code-mixed and monolingual tasks.

## Repository Structure

* `.gitignore`: Standard configuration file specifying files to be excluded from Git version control.
* `LICENSE`: Licensing information governing the use and distribution of this repository's content.
* `Prompts for GPT & LlaMa Models.pdf`: Document detailing the prompts used for fine-tuning GPT-3 and Llama models.

## Code Files Overview

**Fine-tuning Scripts:**

* **`GPT_3_finetuning.py`**: Script for fine-tuning the GPT-3 model on tasks like sentiment analysis and offensive speech prediction.

    This script utilizes the OpenAI API to fine-tune a large language model (LLM) for sentiment analysis. Here's a breakdown of the key steps:

    1. **Data Preparation:**
        - Imports necessary libraries for data manipulation and OpenAI interaction.
        - Reads a CSV file containing tweets and their sentiment labels.
        - Preprocesses the data by:
            - Renaming columns.
            - Stripping whitespaces from tweets.
            - Formatting prompts to include sentiment labels.
        - Saves the preprocessed data as a JSONL file suitable for fine-tuning.
    2. **Fine-tuning:**
        - Uploads the prepared data to OpenAI.
        - Defines variables for data paths and job IDs.
        - Fine-tunes the LLM using OpenAI's API with specified hyperparameters (e.g., number of epochs).
    3. **Using the Fine-tuned Model:**
        - Retrieves the fine-tuned model.
        - Preprocesses test data containing tweets without sentiment labels.
        - Generates sentiment predictions for test tweets using the fine-tuned model's completion functionality.

    **Additional Notes:**
        - Replace placeholders like `csv_file_path` and `test_data_path` with your actual file paths.
        - Set your OpenAI API key before running the script.
        - Some sections are commented out by default (data upload). Uncomment them if needed.

* **`Llama-3.1-8B_finetuning.py`**: This script fine-tunes the Llama-3.1-8B model on various tasks. 
        
    1. **Obtain a Hugging Face API Token:**
       - Create a free Hugging Face account.
       - Go to your profile settings and generate a new token.
       - Save this token as a secure environment variable (e.g., `HF_AUTH_TOKEN`).
    
    2. **Data Preparation:**
       - Load your dataset (e.g., CSV, JSON) containing text data and labels.
       - Preprocess the data:
         - Clean and format the text data.
         - Combine text with labels to create prompts.
         - Split data into training, validation, and test sets.
    3. **Model Initialization:**
       - Load the Llama-3.1-8B model and tokenizer from Hugging Face using your API token:
         ```python
         from transformers import AutoModelForCausalLM, AutoTokenizer
    
         model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",   
     use_auth_token=True)
         tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_auth_token=True)
         ```
    4. **Fine-tuning:**
       - Configure fine-tuning using `transformers` or `peft`.
       - Set hyperparameters (learning rate, batch size, epochs).
       - Train the model and evaluate on a test set.


**Model Training and Translation Scripts:**

* **`Monolingual_model_training.py`**: Trains monolingual models on translated datasets to evaluate their performance on code-mixed tasks.

    This script is designed for training a monolingual model to perform sentiment analysis on translated text. It can be adapted for various languages and tasks by changing the dataset and model name. In our case the code adapts to Sentiment Analysis and Offensive Speech Prediction. Here's a breakdown of the key steps:

    1. **Data Loading and Preprocessing:**
        - Reads sentiment analysis data from a CSV file.
        - Extracts the translated text and corresponding sentiment labels.
        - Splits the data into training, validation, and test sets for robust evaluation.
    2. **Model Selection and Tokenization:**
        - Selects the appropriate monolingual model based on the target language (e.g., Hindi, Tamil, French, Russian).
        - Loads the tokenizer and model from the Hugging Face model hub.
        - Prepares the text data for the model through tokenization.
    3. **Training and Evaluation:**
        - Configures the training process using TrainingArguments, including settings like batch size and learning rate.
        - Initializes a Trainer object to manage the training and evaluation workflows.
        - Trains the model on the training data.
        - Evaluates the model’s performance on the test set.
    4. **Saving Results:**
        - Adds predicted sentiment labels to the test DataFrame.
        - Saves the modified test DataFrame as a CSV file.

    **How to Use:**
        - Clone the repository.
        - Prepare your sentiment analysis dataset in CSV format.
        - Choose the appropriate monolingual model.
        - Run the script: `python Monolingual_model_training.py`.
        - Analyze the results in the saved CSV file.


* **`Translation_codes.py`**: Automates the translation of code-mixed datasets into monolingual formats for training and evaluation.

    This script handles the translation of code-mixed datasets into monolingual text using three different translation methods: GPT-3, Llama-3.1-8B, and Google Translate. The following are the key steps involved:

    1. **Data Loading and Preprocessing:**
        - Loads the code-mixed dataset (e.g., tweets in Hindi-English).
        - Converts the text column to a string format for processing.
        - Handles the dataset for translation by reading from a CSV file.
    2. **GPT-3 Translation:**
        - Uses OpenAI's GPT-3 API to translate code-mixed sentences (e.g., Hinglish) into the target monolingual language (e.g., Hindi).
        - The translation process is automated through a function `translate_code_mixed` that generates translations using a specified prompt.
        - The translated tweets are saved back into the dataset and exported to a new CSV file.
    3. **Llama-3.1-8B Translation:**
        - Utilizes the Llama-3.1-8B model from Hugging Face to translate code-mixed text to the target language.
        - The translation process is adapted to work with different language sets in the research.
        - Text is processed in batches using `translate_text`.
        - The results are saved in output CSV files after each batch is processed.
    4. **Machine Translation (MT) via Google Translate:**
        - Uses the `googletrans` library to detect and translate the code-mixed text.
        - The script identifies the language of the tweet automatically and translates it to the specified target language (e.g., Hindi).
        - Translated tweets are added to the dataset, which is then saved back to the original CSV.

    **How to Use:**
        - Prepare your dataset in CSV format containing code-mixed text.
        - Ensure you have access to the necessary APIs (OpenAI for GPT-3 and Hugging Face for Llama).
        - Run the script to automate translation and save the translated dataset.
        - Example command to run the script: 
        ```bash
        python Translation_codes.py
        ```
        - Check the saved CSV files for the translated tweets.

**BERT Fine-tuning Scripts:**

* **`bert_hs.py` & `bert_sa.py`**:Fine-tunes BERTs in either code-mixed or monolingual datasets

    These scripts fine-tune a BERT-based model for two distinct tasks:
    - **`bert_hs.py`**: Fine-tuning for Offensive Speech Prediction (Hate Speech Detection).
    - **`bert_sa.py`**: Fine-tuning for Sentiment Analysis (SA).

    Both tasks follow similar workflows with the key difference being the dataset and the number of labels for classification:

    1. **Data Loading and Preprocessing:**
        - Load the respective dataset (e.g., CSV, JSON) containing text and labels.
        - Preprocess the text data by:
            - Removing any unnecessary characters or formatting.
            - Tokenizing the text using the BERT tokenizer.
            - Encoding the labels:
                - **For Offensive Speech Prediction**: Labels are typically `toxic/offensive` and `non-toxic/not-offensive` (2 classes).
                - **For Sentiment Analysis**: Labels include `positive`, `neutral`, and `negative` (3 classes).
        - Split the data into training, validation, and test sets.
    
    2. **Model Initialization:**
        - Load the pre-trained BERT model from Hugging Face:
        ```python
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # For Offensive Language
        # OR
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # For Sentiment Analysis
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        ```
        - The model’s tokenizer must align with the model to ensure proper tokenization and encoding.

    3. **Training Configuration:**
        - Set up the training configuration using `TrainingArguments`
        - Configure the optimizer, learning rate, and other hyperparameters as needed.
    
    4. **Model Training:**
        - Use the `Trainer` API to fine-tune the model:
        ```python
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        ```
        - `train_dataset` and `eval_dataset` should be preprocessed text data ready for training.

    5. **Model Evaluation:**
        - After training, the model is evaluated on the test dataset to measure performance, including metrics like accuracy, precision, recall, and F1 score.

    6. **Saving and Using the Model:**
        - Save the trained model for future use:
        ```python
        model.save_pretrained("./trained_bert_model")
        tokenizer.save_pretrained("./trained_bert_tokenizer")
        ```

    **How to Use:**
        - Prepare the dataset in CSV format with text and labels (e.g., `toxic/offensive`, `non-toxic/not-offensive` for OSP or `positive`, `neutral`, `negative` for SA).
        - Ensure the correct model, tokenizer, and dataset paths are set.
        - Run the script:
            - For **Hate Speech Detection**: `python bert_hs.py`
            - For **Sentiment Analysis**: `python bert_sa.py`
        - The model will train and evaluate, and the trained model and tokenizer will be saved in the specified directory.

## Prerequisites

* **Python:** Version 3.7 or later.

* **Dependencies:**
    - The required libraries are specified within the code files.
    - Ensure the following libraries are installed:
        ```bash
        pip install transformers datasets torch scikit-learn googletrans
        ```
    - For fine-tuning tasks using GPT-3, you’ll need access to the OpenAI API and an API key.
    - For Llama-3.1-8B fine-tuning, you need a Hugging Face account and an API token.

* **Hardware Requirements:**
    - A GPU is recommended for efficient model training, especially for fine-tuning large models like GPT-3 or Llama-3.1-8B. 
    - CPUs can also be used, but training will be significantly slower.
    - Ensure you have access to a system with CUDA support if using a GPU.

* **Dataset:**
    - Prepare your datasets in CSV or JSON format for either sentiment analysis or offensive speech prediction. 
    - The dataset should contain the text data and corresponding labels:
        - **For Offensive Speech Prediction**: Labels like `toxic`/`non-toxic`.
        - **For Sentiment Analysis**: Labels like `positive`, `neutral`, or `negative`.
    - The dataset should be appropriately preprocessed, as detailed in each script's comments.

* **Environment Setup:**
    - Set your environment variables for the API tokens:
        - For Hugging Face, set the `HF_AUTH_TOKEN` environment variable with your Hugging Face token.
        - For OpenAI GPT-3 fine-tuning, ensure your OpenAI API key is set in the environment (e.g., `OPENAI_API_KEY`).


* **Hardware Requirements:**
    - The scripts require access to a GPU for efficient training, especially for fine-tuning BERT models. However, they can also run on a CPU, but training will be slower.
    - Recommended: NVIDIA GPU with CUDA support.

* **Dataset:**
    - Ensure you have the correct dataset for either Offensive Speech or Sentiment Analysis tasks, formatted as a CSV or JSON file. The dataset should contain text data and corresponding labels (either `toxic`, `non-toxic` for `bert_hs.py` or `positive`, `neutral`, `negative` for `bert_sa.py`).
