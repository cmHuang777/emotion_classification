"""Run baseline llama3 inference from HuggingFace Inference API

    HF might have speed limit for serverless API. If has speed issue, may try Inference Endpoint: 
    https://huggingface.co/docs/api-inference/parallelism#parallelism-and-batch-jobs
    may be can sleep when rate limit reached
"""

import csv
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("HF_TOKEN")
if access_token is None:
    raise ValueError(
        f"HF access_token is None. Please set up token in system environment."
    )

API_URL = (
    "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
)
headers = {"Authorization": f"Bearer {access_token}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


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
    tokens = generated_text.split()
    for token in tokens:
        if token.lower() in target_labels:
            return token.lower()

    return None


if __name__ == "__main__":
    # predict use case data
    datafile = "data/drone/all_tweets.csv"
    # outfile = datafile.replace("data/", "output/")
    outfile1 = "output/drone/all_tweets_llama3.csv"
    outfile2 = "output/drone/all_tweets_llama3_raw.csv"

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
        counter = 2501

        for row in csv_reader:
            if counter == 0:
                break

            prompt = generate_emotion_and_sentiment_prompt(row["text"])
            llama3_sentiment = None
            llama3_sentiment_raw = None
            llama3_emotion = None
            llama3_emotion_raw = None
            output = None
            while (llama3_sentiment is None or llama3_emotion is None) and (
                (datetime.now() - last_time).total_seconds() < 25
            ):
                while not output and (
                    (datetime.now() - last_time).total_seconds() < 15
                ):
                    output = query({"inputs": prompt, "options": {"use_cache": False}})
                    print(output)
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

                print(output[0]["generated_text"])
                print(llama3_sentiment)
                print(llama3_emotion)

            # # query from server
            # prompt = generate_sentiment_prompt(row["text"])
            # llama3_sentiment = None
            # llama3_sentiment_raw = None
            # # print("Prompt:", prompt)
            # print("Finding sentiment: ")
            # # continue prompting until we get something for sentiment or 20 seconds passed
            # while not llama3_sentiment and ((datetime.now() - last_time).total_seconds() < 20):
            #     output = query({"inputs": prompt, "options": {"use_cache": False}})
            #     llama3_sentiment_raw = output[0]["generated_text"].split(prompt)[-1]
            #     llama3_sentiment = extract_label(llama3_sentiment_raw, target_labels=["positive", "negative", "neutral"])
            #     print(output[0]["generated_text"])
            #     print(llama3_sentiment)

            # prompt = generate_emotion_prompt(row["text"])
            # llama3_emotion = None
            # llama3_emotion_raw = None
            # # print("Prompt:", prompt)
            # print("Finding emotion: ")
            # # continue prompting until we get something for emotion or 20 seconds passed
            # while not llama3_emotion and ((datetime.now() - last_time).total_seconds() < 40):
            #     output = query({"inputs": prompt, "options": {"use_cache": False}})
            #     llama3_emotion_raw = output[0]["generated_text"].split(prompt)[-1]
            #     llama3_emotion = extract_label(llama3_emotion_raw, target_labels=["happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"])
            #     print(output[0]["generated_text"])
            #     print(llama3_emotion)

            row["llama3_sentiment"] = llama3_sentiment
            row["llama3_emotion"] = llama3_emotion
            # print(row)
            csv_writer1.writerow(row)

            row.pop("llama3_sentiment", None)
            row.pop("llama3_emotion", None)
            row["llama3_raw"] = raw
            # row["llama3_sentiment_raw"] = llama3_sentiment_raw
            # row["llama3_emotion_raw"] = llama3_emotion_raw
            csv_writer2.writerow(row)

            t_delta = (datetime.now() - last_time).total_seconds() * 1000
            print("Time elapsed (ms): ", t_delta)
            last_time = datetime.now()

            counter -= 1
            print("======================================")
            # break

        print(f"Total time elapsed (s): {(last_time-start_time).total_seconds()}")
