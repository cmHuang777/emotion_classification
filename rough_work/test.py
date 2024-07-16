import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)

datafile = "data/drone/drone_tweets_qc_annotated.csv"
df = pd.read_csv(datafile, encoding="utf-8", encoding_errors="replace")

emotions = ["happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"]
sentiments = ["positive", "negative", "neutral"]
sent_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
emotion_mapping = {"happiness":0, "anger":1, "disgust":2, "fear":3, "sadness":4, "surprise":5, "other":6}

df["emotion"] = df["golden emotion"].map(emotion_mapping)
df["sentiment"] = df["golden sentiment"].map(sent_mapping)

pred_path = "output/drone/local_llama3_8B/few_shots/masked_all_tweets_llama3.csv"
preds_df = pd.read_csv(pred_path)
preds_df["emotion"] = preds_df["llama3_emotion"].map(emotion_mapping)
preds_df["sentiment"] = preds_df["llama3_sentiment"].map(sent_mapping)

def evaluate(y_true, y_pred):
    # def map_func(x):
    #     return mapping.get(x, 1)
    
    # y_true = np.vectorize(y_true)
    # y_pred = np.vectorize(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, digits=5)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=list(range(len(unique_labels))))
    print('\nConfusion Matrix:')
    print(conf_matrix)

print(df['emotion'].describe())
evaluate(df['emotion'], preds_df['emotion'])
evaluate(df['sentiment'], preds_df['sentiment'])