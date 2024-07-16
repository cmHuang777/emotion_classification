# Sentiment Llama

This repo is based on the private repo provided by my mentor Li Yue. Most of the code is adapted from there. [Link](https://github.com/goPikachu88/SentimentLlama)

# Summary of Prediction Results

(All tests are run on drone tweets dataset)

## Overview

Base model seems to have the best performance, even though the raw data usually don’t give logical explanation and are just some random codes, somehow the predictions match our answers best

## Top Results

### Emotion
- Model: tweet_eval-emotion-1.0-lora-epoch=10-wrong-labels (Fine-tuned Llama)
- Configuration: max_new_toks=30-temp=0.1-rep_pen=1.2-combine_prompts=True-few_shots=True
- weighted avg_f1-score: 0.73504
- Classification Report [here](./output/drone_tweets/tweet_eval-emotion-1.0-lora-epoch=10-wrong-labels/max_new_toks=30-temp=0.1-rep_pen=1.2-combine_prompts=True-few_shots=True/emotion_report.json)

### Sentiment
- Model: Meta-Llama-3-8B-Instruct (Base Llama model)
- Configuration: max_new_toks=30-temp=0.1-rep_pen=1.2-combine_prompts=True-few_shots=True
- weighted avg_f1-score: 0.73623
- Classification Report [here](./output/drone_tweets/Meta-Llama-3-8B-Instruct/max_new_toks=30-temp=0.1-rep_pen=1.2-combine_prompts=True-few_shots=True/sentiment_report.json)


## Other Notes

### Fine tuning on Tweet Eval data
#### Emotion Alone
- the set of emotion labels are different for tweet_eval dataset so 2 approaches were used, 1 using tweet_eval's labels and 1 using drone_tweets's labels
- When trained on drone_tweets's emotion labels (7 emotions), somehow results are not that bad, emotion has highest score when training on 10 epochs
- When trained again with the same set of labels used in tweet_eval dataset (4 emotions):
    - No examples used, no chat template used
    - Results are bad… 

#### Sentiment Alone
- Tweet Eval sentiment training data is too large, about >40k, took only the front 20% (about 10k)
    - Results are not bad, at >0.7 for sentiment
    - Inferior results on emotion, 0.5 for combined prompts, 0.285 for not combining prompts
- Attempted training full training data for 10 epochs:
    - Results are bad, raw is weird, has many repetitions of the predicted label eg “surprise> = surprise> = surprise> = surprise> = surprise> = surprise> = surpri> = surprise> = surpise> =," positive> = positive> = positive> = positive> = positiv...> = positiv...> = positiv..”
    - In comparison, training with 10 epochs of full data on emotion did not give this issue

#### Emotion then sentiment
- Slight error made in the emotion & sentiment prompt that was unnoticed since the start…..
- Used the model previously trained on tweet eval emotion then train on sentiments
- 0.2 sentiment: results are slightly worse, around 0.6 for both emotion and sentiment (both combine and not combining prompts)
- 1.0 sentiment: Results are bad, might be due to the 1.0 sentiment, similar to the case before


### Prompt Engineering
- Placed the convo in a chat template, with start end token and put context into system roles, questions into user roles
    - Performance are generally much lower
    - Raw data are now much more logical explanations of the predictions made, but somehow they may not make sense/don’t match our expectations and may end up giving labels outside of the range provided in the prompt
- May be try using techniques in CARP

### Model Parameters Tuning
- Seem to be random, no clear trend from initial inspection….

### Commands to run the scripts

1. Train / Fine-tune
```bash
python train.py -c training_configs/{config_id}.json
```
(Can also save the terminal output using tee checkpoint/{config_id}.log.txt)

e.g. 
```bash
python train.py -c training_configs/tweet_eval-emotion-1.0-lora.json | tee checkpoint/tweet_eval_emotion-1.0-lora.log.txt
```


2. Predict

Run predictions using a fine-tuned model

```bash
python predict.py \
    --model {model_dir} \
    --test_data {test_path} \
    --output {predictions_out_dir} \
    --max_new_tokens {max_new_tokens} \
    --temperature {temperature} \
    --repetition_penalty {rep_pen} \
    --combine_prompts {combine_prompt} \
    --few_shots {few_shots} \ 
    --include_roles {in_clude_roles} 
```

Run Prediction in batch: modify and run `run_predict_batch.py`


3. Evaluation (Batch)

Generate evaluation report for existing prediction file, in batch & incrementally.

Consolidate existing reports, ranked by dataset name and performance.

```bash
python evaluate.py | tee output/evaluate.log.txt

python consolidate_eval_report.py
```