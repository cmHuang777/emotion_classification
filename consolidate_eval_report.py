import json
from pathlib import Path
import pandas as pd


metrics = {
    "tweet_eval": ["weighted avg", "f1-score"],
    "drone_tweets": ["weighted avg", "f1-score"],
    "drone_reddit": ["weighted avg", "f1-score"],
    "energy_reddit": ["weighted avg", "f1-score"],
}

if __name__ == "__main__":

    emotion_output_filepath = Path("output/consolidated_emotion_report.csv")
    sentiment_output_filepath = Path("output/consolidated_sentiment_report.csv")

    # Generate a new consolidated report based on latest evaluations. Overwrite any old report.
    consolidated_emotion_report_list = []
    consolidated_sentiment_report_list = []
    for dataset_dir in Path("./output").iterdir():
        if dataset_dir.is_dir():
            dataset = dataset_dir.name
            if dataset not in metrics.keys():
                continue

            for model_dir in dataset_dir.iterdir():
                model_id = model_dir.name

                for item in model_dir.iterdir():
                    generation_config = item.name

                    emotion_report_file = Path(item / "emotion_report.json")
                    sentiment_report_file = Path(item / "sentiment_report.json")

                    if emotion_report_file.exists():
                        report = json.load(open(emotion_report_file))
                        primary_metric = metrics[dataset]
                        score = report[primary_metric[0]][primary_metric[1]]

                        print("model_id:", model_id)
                        print("generation_config:", generation_config)
                        print("score:", score)

                        # Overwrite old record if already exists
                        consolidated_emotion_report_list.append(
                            {
                                "dataset": dataset,
                                "model_id": model_id,
                                "generation_config": generation_config,
                                "metric": "_".join(primary_metric),
                                "score": round(score, 5),
                            }
                        )

                    if sentiment_report_file.exists():
                        report = json.load(open(sentiment_report_file))
                        primary_metric = metrics[dataset]
                        score = report[primary_metric[0]][primary_metric[1]]

                        print("model_id:", model_id)
                        print("generation_config:", generation_config)
                        print("score:", score)

                        # Overwrite old record if already exists
                        consolidated_sentiment_report_list.append(
                            {
                                "dataset": dataset,
                                "model_id": model_id,
                                "generation_config": generation_config,
                                "metric": "_".join(primary_metric),
                                "score": round(score, 5),
                            }
                        )

    # Sort the consolidated report
    consolidated_emotion_report_sorted = sorted(
        consolidated_emotion_report_list, key=lambda x: (x["dataset"], -x["score"])
    )
    consolidated_sentiment_report_sorted = sorted(
        consolidated_sentiment_report_list, key=lambda x: (x["dataset"], -x["score"])
    )
    # print(consolidated_report_sorted)

    pd.DataFrame(consolidated_emotion_report_sorted).to_csv(
        emotion_output_filepath,
        columns=["dataset", "model_id", "generation_config", "metric", "score"],
    )
    print(f"Done. Emotion report saved to {emotion_output_filepath}")

    pd.DataFrame(consolidated_sentiment_report_sorted).to_csv(
        sentiment_output_filepath,
        columns=["dataset", "model_id", "generation_config", "metric", "score"],
    )
    print(f"Done. Sentiment report saved to {sentiment_output_filepath}")


"""python consolidate_eval_report.py"""
