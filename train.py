"""
Use padding_side = "right" for training.
Use padding_side = "left" for generation.
"""

import json
import torch

from argparse import ArgumentParser
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from utils import csv_to_dataset, put_in_role_msg

EMOTION_CONTEXT = """
            You are an emotional classifier for online social media text.
            Analyze the emotion of the text enclosed in angle brackets, 
            determine if it is {labels} emotion, and 
            return the answer as the corresponding emotion label {labels}.
        
        """

SENTIMENT_CONTEXT = """
            You are a sentiment classifier for online social media text.
            Analyze the sentiment of the text enclosed in angle brackets, 
            determine if it is {labels}, and 
            return the answer as the corresponding sentiment label {labels}.
        
        """

EMOTION_SENTIMENT_CONTEXT = """
            You are an emotion and sentiment classifier for online social media text.
            Analyze the emotion and sentiment of the text enclosed in angle brackets. 
            For emotion, determine if it is {emotion_labels} emotion.
            For sentiment, determine if it is {sentiment_labels} sentiment.
            Return the answer as "emotion" "sentiment" where emotion is from the corresponding emotion label {emotion_labels}; and sentiment is from the corresponding sentiment label {sentiment_labels};
            emotion followed by sentiment, separated by a space.
            
        """

context_map = {
    "emotion_prompt": EMOTION_CONTEXT,
    "sentiment_prompt": SENTIMENT_CONTEXT,
    "emotion_sentiment_prompt": EMOTION_SENTIMENT_CONTEXT,
}


def generate_context(template, labels):
    labels_str = " or ".join(sorted(list(labels)))
    return template.format_map({"labels": labels_str})


def transform_text(example, text_field, context, include_roles=False):
    prompt = f"<{example['text']}> = "
    output = None
    if include_roles:
        output = put_in_role_msg(context, prompt)
    else:
        output = context + prompt

    example[text_field] = output
    return example


def init_model(model_configs):

    model_name_or_path = model_configs.get("model_dir", None)
    if model_name_or_path is None:
        model_name_or_path = model_configs.get("model_name", None)

    bnb_config = None
    if model_configs.get("quantization", None) and model_configs.get(
        "load_in_4bit", None
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_configs.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=model_configs.get(
                "bnb_4bit_compute_dtype", getattr(torch, "float16")
            ),
            bnb_4bit_quant_type=model_configs.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_configs.get(
                "bnb_4bit_use_double_quant", False
            ),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        quantization_config=bnb_config,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_configs.get("train_padding", "right")

    return model, tokenizer


def init_peft(peft_configs):
    peft_config = None
    # https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py#L88
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(
        "llama", ["q_proj", "v_proj"]
    )

    if peft_configs["enable_peft"] and peft_configs["peft_type"] == "lora":
        peft_config = LoraConfig(
            lora_alpha=configs.get("lora_alpha", 16),
            lora_dropout=configs.get("lora_dropout", 0.1),
            r=configs.get("r", 64),
            bias=configs.get("bias", "none"),
            task_type=configs.get("task_type", "CAUSAL_LM"),
            target_modules=configs.get("target_modules", target_modules),
        )

    return peft_config


def load_training_arguments(training_configs):
    training_arguments = TrainingArguments(
        num_train_epochs=training_configs.get("epoch", 5),
        per_device_train_batch_size=training_configs.get(
            "per_device_train_batch_size", 1
        ),
        gradient_accumulation_steps=training_configs.get(
            "gradient_accumulation_steps", 8
        ),  # 4
        optim=training_configs.get("optimizer", "paged_adamw_32bit"),
        adam_epsilon=training_configs.get("adam_epsilon", 1e-7),
        save_steps=training_configs.get("save_steps", 0),
        logging_steps=training_configs.get("logging_steps", 25),
        learning_rate=training_configs.get("learning_rate", 2e-4),
        weight_decay=training_configs.get("weight_decay", 0.001),
        fp16=training_configs.get("fp16", False),
        bf16=training_configs.get("bf16", True),
        max_grad_norm=training_configs.get("max_grad_norm", 0.3),
        max_steps=training_configs.get("max_steps", -1),
        warmup_ratio=training_configs.get("warmup_ratio", 0.03),
        group_by_length=training_configs.get("group_by_length", True),
        lr_scheduler_type=training_configs.get("lr_scheduler_type", "cosine"),
        report_to=training_configs.get("report_to", "tensorboard"),
        output_dir=training_configs.get("output_dir", "logs"),
        evaluation_strategy=training_configs.get("evaluation_strategy", "epoch"),
    )

    return training_arguments


def train(configs, output_dir):
    # Load data
    # TODO: need to map text to the correct prompt
    # might need to change the label set mentioned in prompt
    train_data = csv_to_dataset(
        configs["dataset"]["train_data"],
        proportion=configs["data_loader"]["training_proportion"],
    )
    eval_data = csv_to_dataset(configs["dataset"]["validation_data"])
    labels = set(train_data["label"])
    labels.update(eval_data["label"])
    text_field = configs["data_loader"]["text_field"]
    context = context_map[text_field]
    context = generate_context(context, labels)
    train_data = train_data.map(
        lambda x: transform_text(x, text_field, context, include_roles=False)
    )
    eval_data = eval_data.map(
        lambda x: transform_text(x, text_field, context, include_roles=False)
    )

    print(f"Fine-tuning with {len(train_data)} data.")
    print(f"Validation with {len(eval_data)} data.")
    max_seq_length = configs["data_loader"].get("max_seq_length", 1024)
    text_field = configs["data_loader"].get("text_field")
    if not text_field:
        raise ValueError("text_field in training config cannot be empty or null")

    # Init model, trainer
    model, tokenizer = init_model(configs["model"])
    peft_config = init_peft(configs["peft"])
    training_arguments = load_training_arguments(configs["training"])
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        dataset_text_field=text_field,
        peft_config=peft_config,
        args=training_arguments,
        packing=False,
        max_seq_length=max_seq_length,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training is done. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate predictions with model")
    parser.add_argument("-c", type=str, required=True, help="Config file path")
    parser.add_argument("-s", type=str, help="Serialization folder")
    args = parser.parse_args()

    configs = json.load(open(args.c))
    config_id = Path(args.c).stem
    output_dir = args.s if args.s else Path("checkpoint/") / config_id

    print(f"Training with config file: {args.c}")
    print("Configurations:\n" + json.dumps(configs, indent=2))
    print(f"Checkpoint will be saved at: {output_dir}")

    train(configs, output_dir)


"""CUDA_VISIBLE_DEVICES=0,1 python train.py -c training_configs/tweet_eval-sentiment-1.0-lora-epoch=10.json | tee checkpoint/tweet_eval-sentiment-1.0-lora-epoch=10/tweet_eval_sentiment.log.txt"""
"""CUDA_VISIBLE_DEVICES=2,3 python train.py -c training_configs/tweet_eval-emotion-sentiment-1.0-lora-epoch=5.json | tee checkpoint/tweet_eval-emotion-sentiment-1.0-lora-epoch=5/tweet_eval_sentiment.log.txt"""
