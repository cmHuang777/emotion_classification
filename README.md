# Sentiment Llama

This repo is based on the private repo provided by my mentor Li Yue. Most of the code is adapted from there. [Link](https://github.com/goPikachu88/SentimentLlama)

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
    --combine_prompts={combine_prompt} \
    --few_shots={few_shots}
```

Run Prediction in batch: modify and run `run_predict_batch.py`


3. Evaluation (Batch)

Generate evaluation report for existing prediction file, in batch & incrementally.

Consolidate existing reports, ranked by dataset name and performance.

```bash
python evaluate.py | tee output/evaluate.log.txt

python consolidate_eval_report.py
```