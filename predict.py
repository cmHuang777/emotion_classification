import os
import sys
import torch

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from peft import PeftModel, PeftConfig
from transformers import (LlamaForCausalLM, BitsAndBytesConfig, HfArgumentParser, pipeline)
from transformers.pipelines.pt_utils import KeyDataset

from args import InferenceConfigs
from utils import (csv_to_dataset, get_dataset_name, init_tokenizer,
                   extract_label, pred_arrays_to_csv, str2bool, transform_text)


def init_model(model_name, cache_dir):
    """
    Initialise the model with the given model_name/path.
    Returns the model object.
    """

    do_quantization = False
    is_adapter_model = False

    compute_dtype = getattr(torch, "bfloat16")  # "bfloat16"
    bnb_config = None

    if Path(f"{model_name}/adapter_config.json").exists():
        is_adapter_model = True

    if is_adapter_model and "4bit" in model_name.name:
        do_quantization = True

    if do_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,  # uses additional quatization to save more ram
        )
        print(f"Model will be quantized.")

    # Load Adapter model
    if is_adapter_model:
        adapter_config = PeftConfig.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=compute_dtype,
            quantization_config=bnb_config,
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            device_map="auto"
        )

    # Load full-paramter model
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=compute_dtype,
            quantization_config=bnb_config,
        )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model.eval()

    return model


def init_pipeline(model, tokenizer, max_new_tokens, temperature=0.6, repetition_penalty=1):
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device_map="auto",
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )

    return pipe


def predict(pipe, ds, combine_prompts=True):
    """
    Predicts using the model and tokenizer, with input dataset and other settings.
    Returns 2 arrays, first one is raw output, 
            second one with extracted emotion and sentiment.
    """

    BATCH_SIZE = 32
    counter = 0
    MAX_ROW = 99999999999  # for partial predictions/testing

    llama3_labels = []
    llama3_raws = []
    start_time = datetime.now()
    last_time = start_time

    if combine_prompts:
        for out in pipe(KeyDataset(ds, "emotion_and_sentiment_prompt"), batch_size=BATCH_SIZE, return_full_text=False):
            if counter >= MAX_ROW:
                break
            t_delta = (datetime.now()-last_time).total_seconds()*1000
            # print("Time elapsed (ms): ", t_delta, " row:", counter)
            # print(out)
            raw = out[0]["generated_text"]
            llama3_sentiment = extract_label(
                raw, target_labels=["positive", "negative", "neutral"])
            llama3_emotion = extract_label(raw, target_labels=[
                                           "happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"])
            llama3_labels.append(
                {"llama3_emotion": llama3_emotion, "llama3_sentiment": llama3_sentiment})
            llama3_raws.append({"llama3_raw": raw})

            last_time = datetime.now()
            counter += 1
        print(f"Total time elapsed generating emotion and sentiment (s): {
              (last_time-start_time).total_seconds()}")

    else:
        # generate the predictions of emotion and sentiment separately
        for out in pipe(KeyDataset(ds, "emotion_prompt"), batch_size=BATCH_SIZE, return_full_text=False):
            if counter >= MAX_ROW:
                break
            t_delta = (datetime.now()-last_time).total_seconds()*1000
            # print("Time elapsed (ms): ", t_delta, " row:", counter)
            # print(out)
            raw = out[0]["generated_text"]
            llama3_emotion = extract_label(raw, target_labels=[
                                           "happiness", "anger", "disgust", "fear", "sadness", "surprise", "other"])
            llama3_labels.append({"llama3_emotion": llama3_emotion})
            llama3_raws.append({"llama3_emotion_raw": raw})

            last_time = datetime.now()
            counter += 1
        print(f"Total time elapsed generating emotion (s): {
              (last_time-start_time).total_seconds()}")

        counter = 0
        for out in pipe(KeyDataset(ds, "sentiment_prompt"), batch_size=BATCH_SIZE, return_full_text=False):
            if counter >= MAX_ROW:
                break
            t_delta = (datetime.now()-last_time).total_seconds()*1000
            # print("Time elapsed (ms): ", t_delta, " row:", counter)
            # print(out)
            raw = out[0]["generated_text"]
            llama3_sentiment = extract_label(
                raw, target_labels=["positive", "negative", "neutral"])
            llama3_labels[counter].update(
                {"llama3_sentiment": llama3_sentiment})
            llama3_raws[counter].update({"llama3_sentiment_raw": raw})

            last_time = datetime.now()
            counter += 1
        print(f"Total time elapsed generating sentiment (s): {
              (last_time-start_time).total_seconds()}")

    return llama3_raws, llama3_labels


if __name__ == "__main__":
    parser = HfArgumentParser(InferenceConfigs)
    configs = None
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (configs,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        (configs,) = parser.parse_args_into_dataclasses()

    model_dir = Path(configs.model)
    model_id = model_dir.name
    test_path = configs.dataset
    max_new_tokens = configs.max_new_tokens
    # deafult temperature=0.6 in generation config
    temperature = configs.temperature
    # default 1 gives no penalty for repetition
    repetition_penalty = configs.repetition_penalty
    include_roles = configs.include_roles
    few_shots = configs.few_shots
    combine_prompts = configs.combine_prompts

    if configs.output_dir:
        output_dir = Path(configs.output_dir)
    else:
        # output_dir = model_dir
        dataset_name = get_dataset_name(test_path)
        output_dir = Path(
            f"output/{dataset_name}/{model_id}/max_new_toks={max_new_tokens}-temp={temperature}"
            f"-rep_pen={repetition_penalty}-combine_prompts={
                combine_prompts}-few_shots={few_shots}"
            f"-include_roles={include_roles}"
        )

    output_dir.mkdir(exist_ok=True, parents=True)
    cache_dir = "cache/" + model_id
    model = init_model(model_dir, cache_dir)
    tokenizer = init_tokenizer(model_dir, cache_dir)
    ds = csv_to_dataset(test_path, few_shots=few_shots)
    ds = ds.map(lambda x: transform_text(
        x, include_roles=include_roles, few_shots=few_shots))
    outfile1 = str(output_dir) + "/raw.csv"
    outfile2 = str(output_dir) + "/predictions.csv"
    # print("outfile1:", outfile1)
    # print("out_dir name:", output_dir)

    pipe = init_pipeline(model, tokenizer, max_new_tokens,
                         temperature, repetition_penalty)
    raws, predictions = predict(pipe, ds, combine_prompts)
    pred_arrays_to_csv(test_path, outfile1, outfile2, raws, predictions)


# CUDA_VISIBLE_DEVICES=5 python predict.py --model meta-llama/Meta-Llama-3-8B-Instruct --test_data data/drone/masked_all_tweets.csv
# CUDA_VISIBLE_DEVICES=5 python predict.py --model checkpoint/tweet_eval-emotion-1.0-lora-epoch=10 --test_data data/drone/masked_all_tweets.csv --repetition_penalty 1.2 --temperature 0.1 --max_new_token 30 --combine_prompts=False
# CUDA_VISIBLE_DEVICES=5 python predict.py ./configs/inference/template.json
