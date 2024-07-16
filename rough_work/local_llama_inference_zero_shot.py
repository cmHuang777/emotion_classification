import os
from string import punctuation
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from datetime import datetime
import csv
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("HF_TOKEN1")
if access_token is None:
    raise ValueError(f"HF access_token is None. Please set up token in system environment.")

	
################################ Start of Utility Functions ################################################

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()

def generate_sentiment_prompt(text):
    return f"""
            Analyze the sentiment of the text enclosed in angle brackets, 
            determine if it is positive, neutral, or negative, and 
            return the answer as the corresponding sentiment label "positive" or "neutral" or "negative".

            <{text}> = """.strip()


def generate_emotion_prompt(text):
    return f"""
            Analyze the emotion of the text enclosed in angle brackets, 
            determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion, and 
            return the answer as the corresponding emotion label "happiness" or "anger" or "disgust" or "fear" or "sadness" or "surprise" or "other".

            <{text}> = """.strip()

def generate_emotion_and_sentiment_prompt(text):
    return f"""
            Analyze the emotion and sentiment of the text enclosed in angle brackets. 
            For emotion, determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion.
            For sentiment, determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion.
            Return the answer as "emotion" "sentiment" where emotion is from the corresponding emotion label "happiness" or "anger" or "disgust" 
            or "fear" or "sadness" or "surprise" or "other"; and sentiment is from the corresponding sentiment label "positive" or "neutral" or "negative"; 
            emotion followed by sentiment, separated by a space.
            
            <{text}> = """.strip()

def extract_label(generated_text, target_labels):
    """Extract from the generated text the first label that is defined in the set of target labels"""
    tokens = generated_text.split()
    for token in tokens:
        token = token.strip(punctuation)
        if token.lower() in target_labels:
            return token.lower()
    
    return None

def predict(model, tokenizer, datafile, outfile1, outfile2):

    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=64,
        # device_map="auto",
        device=1,
    )

    with open(datafile, 'r', newline='') as infile, open(outfile1, 'w', newline='') as out_file1, open(outfile2, 'w', newline='') as out_file2:
        csv_reader = csv.DictReader(infile)
        
        fieldnames1 = csv_reader.fieldnames + ["llama3_sentiment", "llama3_emotion"]
        csv_writer1 = csv.DictWriter(out_file1, fieldnames=fieldnames1)
        csv_writer1.writeheader()

        fieldnames2 = csv_reader.fieldnames + ["llama3_raw"]  # changed from 2 separate columns for emotion and sentiment
        csv_writer2 = csv.DictWriter(out_file2, fieldnames=fieldnames2)
        csv_writer2.writeheader()
        
        start_time = datetime.now()
        last_time = start_time
        counter = 1
        MAX_ROW = 2501
        
        for row in csv_reader:
            # if counter > MAX_ROW: break
            # print("Inferencing row ", counter)
            prompt = generate_emotion_and_sentiment_prompt(row['text'])
            print("Inferencing row", counter)
            output = pipe(prompt)
            # print(f"inference time: {(datetime.now()-last_time).total_seconds()}s")
            llama3_sentiment = None
            llama3_emotion = None

            raw = output[0]["generated_text"].split(prompt)[-1]
            llama3_sentiment = extract_label(raw, target_labels=["positive", "negative", "neutral"])
            llama3_emotion = extract_label(raw, target_labels=["happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"])

            
            # print(output[0]["generated_text"])
            # print(llama3_sentiment)
            # print(llama3_emotion)

            row["llama3_sentiment"] = llama3_sentiment
            row["llama3_emotion"] = llama3_emotion
            # print(row)
            csv_writer1.writerow(row)
            
            row.pop("llama3_sentiment", None)
            row.pop("llama3_emotion", None)
            row["llama3_raw"] = raw
            csv_writer2.writerow(row)

            t_delta = (datetime.now()-last_time).total_seconds()*1000
            print("Time elapsed (ms): ", t_delta)
            last_time = datetime.now()
            
            counter += 1

    print(f"Total time elapsed (s): {(last_time-start_time).total_seconds()}")


########################################## End of Utility Functions ##############################################################

datafile = "data/drone/masked_all_tweets.csv"
outfile1 = "output/drone/local_llama3_8B/zero_shot/masked_all_tweets_llama3.csv"
outfile2 = "output/drone/local_llama3_8B/zero_shot/masked_all_tweets_llama3_raw.csv"
# df = pd.read_csv(filename, encoding="utf-8", encoding_errors="replace")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda"

################# Change this for using diff model ##############
cache_dir = "cache/llama3_8B"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="cuda",
    torch_dtype=compute_dtype,
    token=access_token,
    # quantization_config=bnb_config, 
    cache_dir=cache_dir,
)

# model.to(device)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    token=access_token,
    cache_dir=cache_dir,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 　left for inference

# model, tokenizer = setup_chat_format(model, tokenizer)

predict(model, tokenizer, datafile, outfile1, outfile2)