"""Run baseline llama3 inference from HuggingFace Inference API

    HF might have speed limit for serverless API. If has speed issue, may try Inference Endpoint: 
    https://huggingface.co/docs/api-inference/parallelism#parallelism-and-batch-jobs
    
"""
import csv
import os
import requests
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import json
from string import punctuation

load_dotenv()

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

            For example:
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = anger negative
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = surprise positive
            <The environment can and has survived much hotter conditions.> = other neutral
            
            <{text}> = """.strip()

#### for few shots add in the following to the prompt #####
"""

            For example:
            <You’ve had over a month to get this finalized ! Why are things delayed ?> = anger negative
            <WOW! Drone Delivery Startup, @zipline Raises $25m To Expand Its Operations In Africa> = surprise positive
            <The environment can and has survived much hotter conditions.> = other neutral

"""
def extract_label(generated_text, target_labels):
    """Extract from the generated text the first label that defined in the set of target labels"""
    tokens = generated_text.split()
    for token in tokens:
        token = token.strip(punctuation)
        if token.lower() in target_labels:
            return token.lower()
    
    return None


if __name__ == "__main__":
    time.sleep(7000)
    # predict use case data
    datafile = "data/drone/masked_all_tweets.csv"
    # outfile = datafile.replace("data/", "output/")
    outfile1 = "output/drone/serverless_inferences/few_shots/all_tweets_llama3.csv"
    outfile2 = "output/drone/serverless_inferences/few_shots/all_tweets_llama3_raw.csv"
    token_num = 1
    
    with open(datafile, 'r', newline='') as infile, open(outfile1, 'w', newline='') as out_file1, open(outfile2, 'w', newline='') as out_file2:
        csv_reader = csv.DictReader(infile)
        
        fieldnames1 = csv_reader.fieldnames + ["llama3_sentiment", "llama3_emotion"]
        csv_writer1 = csv.DictWriter(out_file1, fieldnames=fieldnames1)
        csv_writer1.writeheader()

        fieldnames2 = csv_reader.fieldnames + ["llama3_raw"] # merged now that I combine sentiment and emotion for single query/response
        csv_writer2 = csv.DictWriter(out_file2, fieldnames=fieldnames2)
        csv_writer2.writeheader()
        
        start_time = datetime.now()
        last_time = start_time
        counter = 1
        MAX_ROW = 2502
        
        for row in csv_reader:
            # if counter == MAX_ROW: 
            #     break
            # query from server
            # if counter < 930:
            #     counter += 1
            #     continue
            prompt = generate_emotion_and_sentiment_prompt(row["text"])
            # print("Prompt:", prompt)
            print("Inferencing row", counter)
            output = query({"inputs": prompt, "parameters": {"max_new_tokens": 64, "return_full_text": False}}, token_num)
            # print(output)
            while 'error' in output:
                print(f"rate limit reached for token {token_num}!")
                print(output)
                if token_num < 3:
                    token_num += 1
                    print(f"Changing to token {token_num} now")
                    output = query({"inputs": prompt}, token_num)
                else:
                    token_num = 1 
                    now = datetime.now()
                    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    sleep_seconds = (next_hour - now).total_seconds() + 1
                    print("Waiting till next hour and resetting token")
                    time.sleep(sleep_seconds)
                    

            llama3_raw = output[0]["generated_text"].split(prompt)[-1]  # remove the prompt from the generated text if it exists
            llama3_sentiment = extract_label(llama3_raw, target_labels=["positive", "negative", "neutral"])
            llama3_emotion = extract_label(llama3_raw, target_labels=["happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"])

            row["llama3_sentiment"] = llama3_sentiment
            row["llama3_emotion"] = llama3_emotion
            # print(row)
            csv_writer1.writerow(row)
            
            row.pop("llama3_sentiment", None)
            row.pop("llama3_emotion", None)
            row["llama3_raw"] = llama3_raw
            # row["llama3_sentiment_raw"] = llama3_sentiment_raw
            # row["llama3_emotion_raw"] = llama3_emotion_raw
            csv_writer2.writerow(row)
            
            t_delta = (datetime.now()-last_time).total_seconds()*1000
            print("Time elapsed (ms): ", t_delta)
            last_time = datetime.now()
            
            counter += 1
            
        print(f"Total time elapsed (s): {(last_time-start_time).total_seconds()}")
