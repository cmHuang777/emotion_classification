import json
import re
import sys
import pandas as pd
import spacy
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nlp = spacy.load("en_core_web_sm")

test_data_path = {
    "drone_tweets": "data/drone/responses/all_tweets_full_responses.csv",
    "drone_reddit": "data/drone/responses/drone5_all_data.csv",
    "energy_reddit": "data/energy/responses/full_energy.csv"
}

emotion_labels = ["happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"]
sentiment_labels = ["positive", "negative", "neutral"]


def evaluate(y_true, y_pred, labels):
    if not isinstance(y_true, list):
        y_true = y_true.tolist()
        y_pred = y_pred.tolist()

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.5f}')
    
    # Generate accuracy report
    for label in labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.5f}')

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    print('\nConfusion Matrix:')
    print(conf_matrix)        
    
    # Generate classification report
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0.0, digits=5))

    class_report = classification_report(y_true=y_true, y_pred=y_pred, digits=5, labels=labels, 
                                         zero_division=0.0, output_dict=True)
    return class_report


def batch_evaluation():
    for dir in Path("./output").iterdir():
        dataset = dir.name
        if dataset not in test_data_path: 
            continue
        for model in dir.iterdir():
            if model.is_dir():
                model_id = model.name
                # print("model_id:", model_id)
                
                for item in model.iterdir():
                    pred_file = Path(item / "predictions.csv")
                    report_file = Path(item / "report.json")
                                    
                    if pred_file.exists() and not report_file.exists():
                        print("pred_file:", pred_file)
                        # continue
                        
                        # data = [json.loads(line) for line in open(pred_file)]
                        golden = pd.read_csv(test_data_path[dataset])
                        golden = golden[["voted_emotion", "voted_sentiment"]]
                        predictions = pd.read_csv(pred_file)
                        
                        # try:
                        #     report = evaluate(data, labels[dataset])
                        # except:
                        #     print(f"Exception: evaluation failed for {model_id}")
                        #     continue
                        
                        emotion_report = evaluate(golden["voted_emotion"], predictions["llama3_emotion"], emotion_labels)
                        sentiment_report = evaluate(golden["voted_sentiment"], predictions["llama3_sentiment"], sentiment_labels)
                        
                        with open(item / "emotion_report.json", "w") as f2:
                            json.dump(emotion_report, f2, indent=2, ensure_ascii=False)

                        with open(item / "sentiment_report.json", "w") as f2:
                            json.dump(sentiment_report, f2, indent=2, ensure_ascii=False)
                
    return None


if __name__ == "__main__":
    
    print("===== batch evaluations ======")
    batch_evaluation()

"""python evaluate.py | tee output/evaluate.log.txt"""