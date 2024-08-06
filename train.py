"""
Use padding_side = "right" for training.
Use padding_side = "left" for generation.
"""

import logging
import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datasets
import torch
import transformers
from pathlib import Path
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from args import Configs
from utils import csv_to_dataset, init_tokenizer, put_in_role_msg


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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


def transform_text(example, text_field, context, include_roles=False, tokenizer=None):
    prompt = f"<{example['text']}> = "
    output = None
    if include_roles:
        output = put_in_role_msg(context, prompt, tokenizer=tokenizer)
    else:
        output = context + prompt

    example[text_field] = output
    # print(example)
    return example


def init_model(model_configs):
    model_name_or_path = model_configs.get("model_dir", None)
    if model_name_or_path is None:
        model_name_or_path = model_configs.get("model_name_or_path", None)
    do_quantization = model_configs["quantization"]
    is_adapter_model = False
    bnb_config = None

    if Path(f"checkpoint/{model_name_or_path}/adapter_config.json").exists():
        is_adapter_model = True

    if is_adapter_model and "4bit" in model_name_or_path:
        do_quantization = True

    if do_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_configs["load_in_4bit"],
            load_in_8bit=model_configs["load_in_8bit"],
            bnb_4bit_quant_type=model_configs["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(
                torch, model_configs["bnb_4_bit_compute_dtype"]
            ),
            bnb_4bit_use_double_quant=False,  # uses additional quatization to save more ram
        )
        print(f"Model will be quantized.")

    # Load Adapter model
    if is_adapter_model:
        adapter_config = PeftConfig.from_pretrained(model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path,
            cache_dir=model_configs["cache_dir"],
            device_map="auto",
            torch_dtype=getattr(torch, model_configs["compute_dtype"]),
            quantization_config=bnb_config,
        )
        model = PeftModel.from_pretrained(model, model_name_or_path, device_map="auto")

    # Load full-paramter model
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=model_configs["cache_dir"],
            device_map="auto",
            torch_dtype=getattr(torch, model_configs["compute_dtype"]),
            quantization_config=bnb_config,
            token=access_token,
        )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    padding_side = model_configs.get("train_padding", "right")
    tokenizer = init_tokenizer(model_name_or_path, None, padding=padding_side)

    return model, tokenizer


def init_peft(peft_configs):
    peft_config = None
    # https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py#L88
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(
        "llama", ["q_proj", "v_proj"]
    )

    if peft_configs["enable_peft"] and peft_configs["peft_type"] == "lora":
        peft_config = LoraConfig(
            lora_alpha=peft_configs.get("lora_alpha", 16),
            lora_dropout=peft_configs.get("lora_dropout", 0.1),
            r=peft_configs.get("r", 64),
            bias=peft_configs.get("bias", "none"),
            task_type=peft_configs.get("task_type", "CAUSAL_LM"),
            target_modules=peft_configs.get("target_modules", target_modules),
        )

    return peft_config


def train(configs, output_dir):
    model_args = configs.model
    data_args = configs.dataset
    data_loader_args = configs.data_loader
    peft_args = configs.peft
    training_args = configs.training
    train_data = csv_to_dataset(
        data_args["train_file"],
        proportion=data_loader_args["training_proportion"],
    )
    eval_data = csv_to_dataset(data_args["validation_file"])
    labels = set(train_data["label"])
    labels.update(eval_data["label"])
    text_field = data_loader_args["text_field"]
    if not text_field:
        raise ValueError("text_field in training config cannot be empty or null")

    context = context_map[text_field]
    context = generate_context(context, labels)
    print(f"Fine-tuning with {len(train_data)} data.")
    print(f"Validation with {len(eval_data)} data.")

    # init model and tokenizer
    model, tokenizer = init_model(model_args)

    train_data = train_data.map(
        lambda x: transform_text(
            x, text_field, context, include_roles=True, tokenizer=tokenizer
        )
    )
    print(train_data[-1][text_field])
    eval_data = eval_data.map(
        lambda x: transform_text(
            x, text_field, context, include_roles=True, tokenizer=tokenizer
        )
    )

    max_seq_length = data_loader_args.get("max_seq_length", 1024)

    print(train_data[-1][text_field])
    # Init trainer
    peft_config = init_peft(peft_args)
    sft_config = SFTConfig(
        **training_args,
        packing=False,
        max_seq_length=max_seq_length,
        dataset_text_field=text_field,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        args=sft_config,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training is done. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = HfArgumentParser(Configs)
    training_parser = HfArgumentParser(TrainingArguments)
    configs = None
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (configs,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        (configs,) = parser.parse_args_into_dataclasses()

    model_args = configs.model
    data_args = configs.dataset
    data_loader_args = configs.data_loader
    peft_args = configs.peft
    training_args = configs.training
    # trainging_args = training_parser.parse_dict(training_args)

    print("Model Args: \n", model_args)
    print("Data Args: \n", data_args)
    print("Training Args: \n", training_args)
    print("Training arguments classes", training_args.__class__.__name__)
    print("Peft Args: \n", peft_args)
    print("Data Loader Args: \n", data_loader_args)

    # if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.WARNING
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    #     + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    # )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    set_seed(training_args["seed"])

    config_id = Path(sys.argv[1]).stem
    output_dir = Path("checkpoint/") / config_id

    # parser = ArgumentParser(description="Generate predictions with model")
    # parser.add_argument("-c", type=str, required=True, help="Config file path")
    # parser.add_argument("-s", type=str, help="Serialization folder")
    # args = parser.parse_args()

    # configs = json.load(open(args.c))
    # config_id = Path(args.c).stem
    # output_dir = args.s if args.s else Path("checkpoint/") / config_id

    print(f"Training with config file: {sys.argv[1]}")
    # print("Configurations:\n" + json.dumps(configs, indent=2))
    print(f"Checkpoint will be saved at: {output_dir}")

    train(configs, output_dir)


"""CUDA_VISIBLE_DEVICES=0,1 python train.py -c training_configs/tweet_eval-sentiment-1.0-lora-epoch=10.json | tee checkpoint/tweet_eval-sentiment-1.0-lora-epoch=10/tweet_eval_sentiment.log.txt"""
"""CUDA_VISIBLE_DEVICES=2,3 python train.py -c training_configs/tweet_eval-emotion-sentiment-1.0-lora-epoch=5.json | tee checkpoint/tweet_eval-emotion-sentiment-1.0-lora-epoch=5/tweet_eval_sentiment.log.txt"""
"""CUDA_VISIBLE_DEVICES=3,4,5 python train.py ./configs/train/tweet_eval-emotion-1.0-lora-epoch=3.json"""
