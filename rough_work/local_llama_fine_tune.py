import os
from string import punctuation

import spacy

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import csv
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("HF_TOKEN1")
if access_token is None:
    raise ValueError(
        f"HF access_token is None. Please set up token in system environment."
    )

################################ Start of Utility Functions ################################################

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()


def generate_sentiment_prompt(text):
    return f"""
            Analyze the sentiment of the text enclosed in angle brackets, 
            determine if it is positive, neutral, or negative, and 
            return the answer as the corresponding sentiment label "positive" or "neutral" or "negative".
            For example: 
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = negative
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = positive
            <The environment can and has survived much hotter conditions.> = neutral

            <{text}> = """.strip()


def generate_emotion_prompt(text):
    return f"""
            Analyze the emotion of the text enclosed in angle brackets, 
            determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion, and 
            return the answer as the corresponding emotion label "happiness" or "anger" or "disgust" or "fear" or "sadness" or "surprise" or "other".
            For example:
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = anger
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = surprise
            <The environment can and has survived much hotter conditions.> = other

            <{text}> = """.strip()


def generate_emotion_and_sentiment_prompt(text):
    return f"""
            Analyze the emotion and sentiment of the text enclosed in angle brackets. 
            For emotion, determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion.
            For sentiment, determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion.
            Return the answer as "emotion" "sentiment" where emotion is from the corresponding emotion label "happiness" or "anger" or "disgust" 
            or "fear" or "sadness" or "surprise" or "other"; and sentiment is from the corresponding sentiment label "positive" or "neutral" or "negative"; 
            emotion followed by sentiment, separated by a space.
            
            For example:
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = anger negative
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = surprise positive
            <The environment can and has survived much hotter conditions.> = other neutral

            <{text}> = """.strip()


def extract_label(generated_text, target_labels):
    """Extract from the generated text the first label that is defined in the set of target labels"""
    nlp = spacy.load("en_core_web_sm")
    tokens = [token.text for token in nlp(generated_text)]
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
        # device=2,
        device_map="auto",
        # padding=True,
    )

    with open(datafile, "r", newline="") as infile, open(
        outfile1, "w", newline=""
    ) as out_file1, open(outfile2, "w", newline="") as out_file2:
        csv_reader = csv.DictReader(infile)

        fieldnames1 = csv_reader.fieldnames + ["llama3_sentiment", "llama3_emotion"]
        csv_writer1 = csv.DictWriter(out_file1, fieldnames=fieldnames1)
        csv_writer1.writeheader()

        fieldnames2 = csv_reader.fieldnames + [
            "llama3_raw"
        ]  # changed from 2 separate columns for emotion and sentiment
        csv_writer2 = csv.DictWriter(out_file2, fieldnames=fieldnames2)
        csv_writer2.writeheader()

        start_time = datetime.now()
        last_time = start_time
        counter = 1
        MAX_ROW = 2

        for row in csv_reader:
            if counter > MAX_ROW:
                break
            prompt = generate_emotion_and_sentiment_prompt(row["text"])

            print("Inferencing row", counter)
            # print("tokenized input:", tokenizer(prompt, padding="max_length", max_length=512))
            output = pipe(prompt)
            # print(f"inference time: {(datetime.now()-last_time).total_seconds()}s")
            llama3_sentiment = None
            llama3_emotion = None

            raw = output[0]["generated_text"].split(prompt)[-1]
            llama3_sentiment = extract_label(
                raw, target_labels=["positive", "negative", "neutral"]
            )
            llama3_emotion = extract_label(
                raw,
                target_labels=[
                    "happiness",
                    "anger",
                    "disgust",
                    "fear",
                    "sadness",
                    "surprise",
                    "other",
                ],
            )

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

            t_delta = (datetime.now() - last_time).total_seconds() * 1000
            print("Time elapsed (ms): ", t_delta)
            last_time = datetime.now()

            counter += 1

    print(f"Total time elapsed (s): {(last_time-start_time).total_seconds()}")


def predict(model, tokenizer, df):
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        # device=2,
        device_map="auto",
        # padding=True,
    )
    output = pd.DataFrame(columns=["emotion", "sentiment"])
    sentiments = []
    emotions = []
    for _, row in df.iterrows():
        prompt = row["text"]
        output = pipe(prompt)
        raw = output[0]["generated_text"].split(prompt)[-1]
        llama3_sentiment = extract_label(
            raw, target_labels=["positive", "negative", "neutral"]
        )
        llama3_emotion = extract_label(
            raw,
            target_labels=[
                "happiness",
                "anger",
                "disgust",
                "fear",
                "sadness",
                "surprise",
                "other",
            ],
        )
        sentiments.append(llama3_sentiment)
        emotions.append(llama3_emotion)

    output = pd.DataFrame({"emotion": emotions, "sentiment": sentiments})

    # print(output.describe())
    return output


# def map_func(x):
#     return mapping.get(x, 1)


def evaluate(y_true, y_pred, labels):

    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f"Accuracy: {accuracy:.5f}")

    for label in labels:
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f"Accuracy for label {label}: {accuracy:.5f}")

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, digits=5)
    print("\nClassification Report:")
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)


def prepare_dataset(input_csv, label_str):
    # converts csv file to huggingface dataset object with train test split stratified
    df = pd.read_csv(input_csv, encoding="utf-8", encoding_errors="replace")
    df["emotion"] = df["voted_emotion"]
    df["sentiment"] = df["voted_sentiment"]
    df = df[["text", "emotion", "sentiment"]]
    if label_str == "emotion":
        df["text"] = df["text"].map(generate_emotion_prompt)
    elif label_str == "sentiment":
        df["text"] = df["text"].map(generate_sentiment_prompt)
    else:
        raise ValueError(
            f"Label string in prepare dataset method should only be either emotion or \
                         sentiment but label_str={label_str} is provided"
        )
    emotion_df = df["emotion"]
    df = df.rename(columns={label_str: "label"})
    # ensure that the train test split is the same across all ds, to ensure fair comparison
    df_train, df_test = train_test_split(
        df, test_size=0.7, random_state=88, stratify=emotion_df
    )
    ds_train = Dataset.from_pandas(df_train).remove_columns("__index_level_0__")
    ds_test = Dataset.from_pandas(df_test).remove_columns("__index_level_0__")
    ds = DatasetDict({"train": ds_train, "validation": ds_test, "test": ds_test})
    return ds


########################################## End of Utility Functions ##############################################################


# datafile = "data/drone/masked_all_tweets.csv"
# outfile1 = "output/drone/local_llama3_8B/test/masked_all_tweets_llama3.csv"
# outfile2 = "output/drone/local_llama3_8B/test/masked_all_tweets_llama3_raw.csv"
datafile = "data/drone/responses/all_tweets_full_responses.csv"
emotion_ds = prepare_dataset(datafile, "emotion")
sentiment_ds = prepare_dataset(datafile, "sentiment")
# df = pd.read_csv(datafile, encoding="utf-8", encoding_errors="replace")

emotions = ["happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"]
sentiments = ["positive", "negative", "neutral"]
sent_mapping = {"positive": 2, "neutral": 1, "negative": 0}
emotion_mapping = {
    "happiness": 0,
    "anger": 1,
    "disgust": 2,
    "fear": 3,
    "sadness": 4,
    "surprise": 5,
    "other": 6,
}

# df["emotion"] = df["voted_emotion"]
# df["sentiment"] = df["voted_sentiment"]
# df = df[["text","emotion","sentiment"]]
# df["text"] = df["text"].map(generate_emotion_and_sentiment_prompt)
# # ensure test set has same distribution as full dataset

# df_train, df_test = train_test_split(df, test_size=0.15, random_state=88, stratify=df['emotion'])

# emotion_df = df_train.rename(columns={"emotion": "label"})
# sentiment_df = df_train.rename(columns={"sentiment": "label"})

# emotion_dataset = Dataset.from_pandas(emotion_df).remove_columns("__index_level_0__")
# sentiment_dataset = Dataset.from_pandas(sentiment_df).remove_columns("__index_level_0__")


# pred_path = "output/drone/local_llama3_8B/few_shots/masked_all_tweets_llama3.csv"
# preds_df = pd.read_csv(pred_path)
# preds_df["emotion"] = preds_df["llama3_emotion"]
# preds_df["sentiment"] = preds_df["llama3_sentiment"]

# emotion_dataset = emotion_dataset.train_test_split(test_size=0.2, seed=88, stratify_by_column="emotion")
# sentiment_dataset = sentiment_dataset.train_test_split(test_size=0.2, seed=88, stratify_by_column="sentiment")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda"

cache_dir = "cache/llama3_70B"
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

compute_dtype = getattr(torch, "bfloat16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=compute_dtype,
    token=access_token,
    quantization_config=bnb_config,
    cache_dir=cache_dir,
)

model = prepare_model_for_kbit_training(model=model)

# model.to(device)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.train()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=access_token,
    cache_dir=cache_dir,
    # padding=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 　left for inference

output_dir = "output/drone/local_llama3_70B/trained_weigths"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=output_dir,  # directory to save and repository id
    num_train_epochs=2,  # number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    adam_epsilon=1e-7,
    save_steps=0,
    logging_steps=10,  # log every 10 steps
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    group_by_length=True,
    lr_scheduler_type="cosine",  # use cosine learning rate scheduler
    # report_to="tensorboard",                  # report metrics to tensorboard
    eval_strategy="epoch",  # save checkpoint every epoch
)

emotion_trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=emotion_ds["train"],
    eval_dataset=emotion_ds["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=1024,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
)

sentiment_trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=sentiment_ds["train"],
    eval_dataset=sentiment_ds["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=1024,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
)

# model, tokenizer = setup_chat_format(model, tokenizer)
# predict(model, tokenizer, datafile, outfile1, outfile2)

emotion_trainer.train()
emotion_output_dir = output_dir + "/emotion"
emotion_trainer.save_model(emotion_output_dir)
tokenizer.save_pretrained(output_dir)

sentiment_trainer.train()
sentiment_output_dir = output_dir + "/sentiment"
sentiment_trainer.save_model(sentiment_output_dir)
# tokenizer.save_pretrained(sentiment_output_dir)

import gc

del [
    model,
    tokenizer,
    peft_config,
    emotion_trainer,
    sentiment_trainer,
    bnb_config,
    training_arguments,
]
# del [df]
del [TrainingArguments, SFTTrainer, LoraConfig, BitsAndBytesConfig]

for _ in range(100):
    torch.cuda.empty_cache()
    gc.collect()

from peft import AutoPeftModelForCausalLM

compute_dtype = getattr(torch, "float16")
tokenizer = AutoTokenizer.from_pretrained(output_dir)
finetuned_emotion_model = emotion_output_dir
finetuned_sentiment_model = sentiment_output_dir

emotion_model = AutoPeftModelForCausalLM.from_pretrained(
    finetuned_emotion_model,
    torch_dtype=compute_dtype,
    return_dict=False,
    low_cpu_mem_usage=True,
    device_map=device,
)

merged_model = emotion_model.merge_and_unload()
merged_model.save_pretrained(
    "./finetuned_llama3_70B/emotion", safe_serialization=True, max_shard_size="2GB"
)
tokenizer.save_pretrained("./finetuned_llama3_70B")

emotion_df_test = pd.DataFrame(emotion_ds["test"])
emotion_preds = predict(merged_model, tokenizer, emotion_df_test)
evaluate(emotion_df_test["label"], emotion_preds["emotion"], emotions)

sentiment_model = AutoPeftModelForCausalLM.from_pretrained(
    finetuned_sentiment_model,
    torch_dtype=compute_dtype,
    return_dict=False,
    low_cpu_mem_usage=True,
    device_map=device,
)

merged_model = sentiment_model.merge_and_unload()
merged_model.save_pretrained(
    "./finetuned_llama3_70B/sentiment", safe_serialization=True, max_shard_size="2GB"
)

sentiment_df_test = pd.DataFrame(sentiment_ds["test"])
sentiment_preds = predict(merged_model, tokenizer, sentiment_df_test)
evaluate(sentiment_df_test["label"], sentiment_preds["sentiment"], sentiments)
