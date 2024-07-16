import json
from string import punctuation
import time
import requests
import spacy
import pandas as pd
import os
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import csv
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("HF_TOKEN1")
if access_token is None:
    raise ValueError(f"HF access_token is None. Please set up token in system environment.")

global_tokenizer = None

# query with the payload using the specific token number
# this way we can query with different token when limit is reached :)
def query(payload, token_num):
    access_token = os.getenv(f"HF_TOKEN{token_num}")
    if access_token is None:
        raise ValueError(f"HF access_token is None. Please set up token in system environment.")
    
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {access_token}"}
    sleep_time = 1 # variable sleep time if this query does not work
    # handles errors until we get a result
    while True:
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        except json.decoder.JSONDecodeError:
            print("JSONDecodeError: Expecting value. Retrying...")
            time.sleep(sleep_time)
            sleep_time *= 2
        except Exception as err:
            print(f"An error occurred: {err}")
            time.sleep(sleep_time)
            sleep_time *= 2

EMOTION_CONTEXT = """
            You are an emotional classifier for online social media text.
            Analyze the emotion of the text enclosed in angle brackets, 
            determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion, and 
            return the answer as the corresponding emotion label "happiness" or "anger" or "disgust" or "fear" or "sadness" or "surprise" or "other".
        
        """

SENTIMENT_CONTEXT = """
            You are a sentiment classifier for online social media text.
            Analyze the sentiment of the text enclosed in angle brackets, 
            determine if it is positive, neutral, or negative, and 
            return the answer as the corresponding sentiment label "positive" or "neutral" or "negative".
        
        """

EMOTION_SENTIMENT_CONTEXT = """
            You are an emotion and sentiment classifier for online social media text.
            Analyze the emotion and sentiment of the text enclosed in angle brackets. 
            For emotion, determine if it is happiness, anger, disgust, fear, sadness, surprise or other emotion.
            For sentiment, determine if it is positive, neutral, or negative.
            Return the answer as "emotion" "sentiment" where emotion is from the corresponding emotion label "happiness" or "anger" or "disgust" 
            or "fear" or "sadness" or "surprise" or "other"; and sentiment is from the corresponding sentiment label "positive" or "neutral" or "negative"; 
            emotion followed by sentiment, separated by a space.
            
            """

EMOTION_EXAMPLES = """
            For example:
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = anger
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = surprise
            <The environment can and has survived much hotter conditions.> = other
        """

SENTIMENT_EXAMPLES = """
            For example: 
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = negative
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = positive
            <The environment can and has survived much hotter conditions.> = neutral
        
        """

EMOTION_SENTIMENT_EXAMPLES = """
            For example:
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = anger negative
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = surprise positive
            <The environment can and has survived much hotter conditions.> = other neutral

        """

EMOTION_CONTEXT_WITH_EXAMPLES = EMOTION_CONTEXT + EMOTION_EXAMPLES
SENTIMENT_CONTEXT_WITH_EXAMPLES = SENTIMENT_CONTEXT + SENTIMENT_EXAMPLES
EMOTION_SENTIMENT_CONTEXT_WITH_EXAMPLES = EMOTION_SENTIMENT_CONTEXT + EMOTION_SENTIMENT_EXAMPLES

def generate_sentiment_prompt(text, few_shots=True):
    prompt = SENTIMENT_CONTEXT
    if few_shots:
        prompt = SENTIMENT_CONTEXT_WITH_EXAMPLES
    prompt += f"<{text}> = ".strip()
    return prompt


def generate_emotion_prompt(text, few_shots=True):
    prompt = EMOTION_CONTEXT
    if few_shots:
        prompt = EMOTION_CONTEXT_WITH_EXAMPLES
    prompt += f"<{text}> = ".strip()
    return prompt

def generate_emotion_and_sentiment_prompt(text, few_shots=True):
    prompt = EMOTION_SENTIMENT_CONTEXT
    if few_shots:
        prompt = EMOTION_SENTIMENT_CONTEXT_WITH_EXAMPLES
    prompt += f"<{text}> = ".strip()
    return prompt

def put_in_role_msg(system_content, prompt):
    """
    Put prompt into the system-user role based format. Resultant response from LLM will be in similar format
    To extract the reponses, do output[0]["generated_text"][-1]["content"]
    """
    message = [
        # possibly can add in context here? and ask for reasoning too
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    # tokenizer = init_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct", "cache/Meta-Llama-3-8B-Instruct")
    # print(tokenizer.default_chat_template)
    message = global_tokenizer.apply_chat_template(message, add_generation_prompt=True)

    return message

def extract_label(generated_text, target_labels):
    """Extract from the generated text the first label that is defined in the set of target labels"""
    nlp = spacy.load("en_core_web_sm")
    tokens = [token.text for token in nlp(generated_text)]
    for token in tokens:
        token = token.strip(punctuation)
        if token.lower() in target_labels:
            return token.lower()
    
    return None

# deprecated, needs updates
def predict(model, tokenizer, datafile, outfile1, outfile2):
    """
    Predicts using the model and tokenizer, on input dataset datafile and 
    write the raw output into outfile1, refined/extracted results into outfile2.
    """
    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=64,
        # device=2,
        device_map="auto",
        # padding=True,
        temperature=0.3,
        repetition_penalty=1.2
    )

    counter = 1
    MAX_ROW = 999999999

    dataset = load_dataset('csv', data_files=datafile, split="train")
    def transform_text(example):
        example['text'] = generate_emotion_and_sentiment_prompt(example['text'])
        return example
    prompts = dataset.map(transform_text)
    # print(dataset)
    llama3_labels = []
    llama3_raws = []
    start_time = datetime.now()
    last_time = start_time
    for out in pipe(KeyDataset(prompts, "text"), batch_size=32, return_full_text=False):
        # if counter > MAX_ROW: break
        t_delta = (datetime.now()-last_time).total_seconds()*1000
        print("Time elapsed (ms): ", t_delta)
        # print(out)
        raw = out[0]["generated_text"]
        llama3_sentiment = extract_label(raw, target_labels=["positive", "negative", "neutral"])
        llama3_emotion = extract_label(raw, target_labels=["happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"])
        llama3_labels.append({"llama3_emotion": llama3_emotion, "llama3_sentiment": llama3_sentiment})
        llama3_raws.append({"llama3_raw": raw})

        last_time = datetime.now()
        counter += 1
    print(f"Total time elapsed (s): {(last_time-start_time).total_seconds()}")

    with open(datafile, 'r', newline='') as infile, open(outfile1, 'w', newline='') as out_file1, open(outfile2, 'w', newline='') as out_file2:
        csv_reader = csv.DictReader(infile)
        
        fieldnames1 = csv_reader.fieldnames + ["llama3_sentiment", "llama3_emotion"]
        csv_writer1 = csv.DictWriter(out_file1, fieldnames=fieldnames1)
        csv_writer1.writeheader()

        fieldnames2 = csv_reader.fieldnames + ["llama3_raw"]  # changed from 2 separate columns for emotion and sentiment
        csv_writer2 = csv.DictWriter(out_file2, fieldnames=fieldnames2)
        csv_writer2.writeheader()

        ds_list = dataset.to_list()
        for i in range(len(llama3_labels)):
            llama3_labels[i].update(ds_list[i])
            llama3_raws[i].update(ds_list[i])

        csv_writer1.writerows(llama3_labels)
        csv_writer2.writerows(llama3_raws)
        print("Finished writing")



def pred_arrays_to_csv(data_file, outfile1, outfile2, raws, labels) -> None:
    """
    Write raw and label arrays generated from prediction function into separate csv files.
    Parameters
    outfile1: the path to write the raw output from LLM 
    outfile2: the path to write the extracted labels (emotions and sentiments)
    """
    with open(data_file, 'r', newline='') as infile, open(outfile1, 'w', newline='') as out_file1, open(outfile2, 'w', newline='') as out_file2:
        csv_reader = csv.DictReader(infile)
        
        fieldnames1 = csv_reader.fieldnames + list(raws[0].keys())  # columns depends on whether emotion and sentiment prompts are merged
        csv_writer1 = csv.DictWriter(out_file1, fieldnames=fieldnames1)
        csv_writer1.writeheader()

        fieldnames2 = csv_reader.fieldnames + ["llama3_sentiment", "llama3_emotion"]
        csv_writer2 = csv.DictWriter(out_file2, fieldnames=fieldnames2)
        csv_writer2.writeheader()

        input_data_dicts = list(csv_reader)
        for i in range(len(labels)):
            labels[i].update(input_data_dicts[i])
            raws[i].update(input_data_dicts[i])

        csv_writer1.writerows(raws)
        csv_writer2.writerows(labels)
        print("Finished writing")

def prepare_dataset(input_path, label_str):
    """
    Prepares dataset into train/validation/test. If input_path is a csv file, does manual splitting.
    Else, loads the dataset directly from huggingface. 

    Parameters
    input_path: the dataset to be loaded/or the csv file to be processed
    returns: Huggingface DatasetDict object containing train, validation, test splits
    """
    # check if the input dataset on huggingface
    if '.csv' not in input_path:
        # attempt load dataset from hugginface
        ds = load_dataset(input_path)
        return ds

    # converts csv file to huggingface dataset object with train test split stratified on emotion
    df = pd.read_csv(input_path, encoding="utf-8", encoding_errors="replace")
    df["emotion"] = df["voted_emotion"]
    df["sentiment"] = df["voted_sentiment"]
    df = df[["text","emotion","sentiment"]]
    if label_str == 'emotion':
        df['text'] = df['text'].map(generate_emotion_prompt)
    elif label_str == 'sentiment':
        df['text'] = df['text'].map(generate_sentiment_prompt)
    else:
        raise ValueError(f"Label string in prepare dataset method should only be either emotion or \
                         sentiment but label_str={label_str} is provided")
    emotion_df = df['emotion']
    df = df.rename(columns={label_str: "label"})
    # ensure that the train test split is the same across all ds, to ensure fair comparison
    df_train, df_test = train_test_split(df, test_size=0.7, random_state=88, stratify=emotion_df)
    ds_train = Dataset.from_pandas(df_train).remove_columns("__index_level_0__")
    ds_test = Dataset.from_pandas(df_test).remove_columns("__index_level_0__")
    ds = DatasetDict({
        'train': ds_train,
        'validation': ds_test,
        'test': ds_test
    })
    return ds

def transform_text(example, include_roles=False, few_shots=True):
    if include_roles:
        user_msg = f"<{example['text']}> = ".strip()
        if few_shots:
            example["emotion_prompt"] = put_in_role_msg(EMOTION_CONTEXT_WITH_EXAMPLES, user_msg)
            example["sentiment_prompt"] = put_in_role_msg(SENTIMENT_CONTEXT_WITH_EXAMPLES, user_msg)
            example["emotion_and_sentiment_prompt"] = put_in_role_msg(EMOTION_SENTIMENT_CONTEXT_WITH_EXAMPLES, user_msg)
        else:
            example["emotion_prompt"] = put_in_role_msg(EMOTION_CONTEXT, user_msg)
            example["sentiment_prompt"] = put_in_role_msg(SENTIMENT_CONTEXT, user_msg)
            example["emotion_and_sentiment_prompt"] = put_in_role_msg(EMOTION_SENTIMENT_CONTEXT, user_msg)
    else:
        example["emotion_prompt"] = generate_emotion_prompt(example['text'], few_shots=few_shots)
        example['sentiment_prompt'] = generate_sentiment_prompt(example['text'], few_shots=few_shots)
        example["emotion_and_sentiment_prompt"] = generate_emotion_and_sentiment_prompt(example['text'], few_shots=few_shots)
    return example

def csv_to_dataset(file_path, proportion=1, few_shots=True):
    """
    converts csv file to dataset, with prompt mapped for each text utterance.
    If proportion is given, select that proportion of the dataset from the front.
    """
    dataset = load_dataset('csv', data_files=file_path, split="train")

    if proportion < 1:
        data_size = int(len(dataset) * proportion)
        dataset = dataset[:data_size]
        dataset = Dataset.from_pandas(pd.DataFrame(data=dataset))
        
    print(f"Loaded {len(dataset)} data. Proportion={proportion}")
    return dataset



def init_tokenizer(model_name, cache_dir, padding="left"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        token=access_token,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding  # 　left for inference, right for training

    global_tokenizer = tokenizer

    return tokenizer

def get_dataset_name(dataset_path):
    return dataset_path.split("/")[-1].split(".")[0]


# for parsing boolean values in predict function
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')