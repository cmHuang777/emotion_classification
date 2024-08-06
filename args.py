from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import TrainingArguments

from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers.utils.versions import require_version


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    train_padding: Optional[str] = field(default="right")
    infer_padding: Optional[str] = field(default="left")
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    quantization: bool = field(
        default=False, metadata={"help": "Whether to use quantization on the model."}
    )
    load_int_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to load the model in 4 bit quantization."},
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={
            "help": "This sets the computational type which might be different than the input type."
        },
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={
            "help": "This sets the quantization data type in the bnb.nn.Linear4Bit layers."
        },
    )
    bnb_4bit_use_double_quant: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use nested quantization to save even more space."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


# https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py#L88
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(
    "llama", ["q_proj", "v_proj"]
)


@dataclass
class PeftArguments:
    enable_peft: bool = field(default=True)
    peft_type: Optional[str] = field(default="lora")
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    r: int = field(default=64)
    bias: str = field(default="none")
    task_type: str = field(default="CAUSAL_LM")
    target_modules: list[str] = field(default_factory=target_modules)


@dataclass
class DataLoaderArguments:
    text_field: str = field(default="prompt")
    training_proportion: float = field(default=1.0)
    max_seq_length: int = field(default=1024)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`train_file` should be a csv, a json or a txt file."
                    )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`validation_file` should be a csv, a json or a txt file."
                    )


@dataclass
class Configs:
    dataset: Dict[str, str]
    data_loader: DataLoaderArguments
    model: ModelArguments
    peft: PeftArguments
    training: TrainingArguments


@dataclass
class InferenceConfigs:
    dataset: str = field(
        default=None,
        metadata={"help": "Test data path"},
    )
    model: str = field(
        default=None,
        metadata={"help": "Model serialization folder"},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Output folder"},
    )
    max_new_tokens: int = field(
        default=50,
        metadata={"help": "Max number of new generated tokens"},
    )
    temperature: float = field(
        default=0.6,
        metadata={"help": "Temperature for text generation"},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalty for repeated text in text generated"},
    )
    include_roles: bool = field(
        default=False,
        metadata={"help": "Whether to use system-user roles format for prompting"},
    )
    few_shots: int = field(
        default=0,
        metadata={"help": "How many examples in the prompts"},
    )
    combine_prompts: bool = field(
        default=True,
        metadata={
            "help": "Whether to combine emotion and sentiment prompt as a single prompt"
        },
    )
