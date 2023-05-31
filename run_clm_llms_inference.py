#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers import AutoConfig, AutoTokenizer, AutoModel
from llm_trainer import LLMTrainer, inference_generation, batch_inference_generation
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# xxx: 2023-03-21
import copy

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, LlamaConfig, WhisperConfig, WhisperModel, LlamaModel, LlamaTokenizer
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import numpy as np
import json
import pickle
import codecs
from PIL import Image
from peft import PeftModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm, trange
from sklearn.metrics import top_k_accuracy_score
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from modeling import MM_LLMs, MM_LLMs_Config
import clip
import whisper

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# xxx: 2023-03-21
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis

def load_datasets(data_args):
    from datasets.dataset_dict import DatasetDict
    from datasets import Dataset

    data_dir = data_args.train_file
    video_names = ["data/avsd/train_video_names.json", "data/vqa/vqa_video_names.json"]
    all_train_dataset = pickle.load(
        open(data_dir, 'rb'))
    print(type(all_train_dataset))
    for k in all_train_dataset:
        print(k, all_train_dataset[k][0])

    # labels = copy.deepcopy(all_train_dataset["input_ids"])
    # all_train_dataset['labels'] = labels

    pad_token_id = 32006
    all_train_dataset['labels'] = [[(l if l != pad_token_id else IGNORE_INDEX) for l in label] 
    for label in all_train_dataset['labels']]

    vname = json_load(video_names[0])['data']
    vname.extend(json_load(video_names[1])['data'])
    visual_data_names = {'data': vname}

    all_train_dataset['images'] = [[e] for e in all_train_dataset['images']]
    all_train_dataset['audios'] = [[e] for e in all_train_dataset['audios']]
    all_train_dataset['videos'] = [[e] for e in all_train_dataset['videos']]


    # all_train_dataset = TensorDataset(all_train_dataset['images'], all_train_dataset['audios'], 
    # all_train_dataset['videos'], all_train_dataset['input_ids'], all_train_dataset['labels'])

    eval_offset = 200
    train_dataset = {'train': Dataset.from_dict({k: all_train_dataset[k] for k in all_train_dataset})}

    train_dataset = DatasetDict(train_dataset)
    all_train_dataset = (train_dataset['train'], visual_data_names)

    return all_train_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # training_args.device = torch.device('cpu')
    # print(training_args)

    training_args.remove_unused_columns=False
    tokenizer = LlamaTokenizer.from_pretrained('trained_models/llama_tokenizer')
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    # tokenizer.padding_side = "right"

    # # xxx: 2023-03-21, add special tokens
    # tokenizer.add_special_tokens(
    #     {
    #         "eos_token": DEFAULT_EOS_TOKEN,
    #         "bos_token": DEFAULT_BOS_TOKEN,
    #         "unk_token": DEFAULT_UNK_TOKEN,
    #     }
    # )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Set seed before initializing model.
    set_seed(training_args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # load model
    clip_config = CLIPConfig.from_pretrained('trained_models/clip_model')
    whisper_config = WhisperConfig.from_pretrained('trained_models/whisper_model')
    llm_config = AutoConfig.from_pretrained('trained_models/llama_model')

    model_config = MM_LLMs_Config(n_frames=6, attention_heads=8, clip_config=clip_config, whisper_config=whisper_config, llm_config=llm_config)

    model = MM_LLMs.from_pretrained(
        training_args.output_dir,
        config = model_config,
        # load_in_8bit=True,
        # torch_dtype=torch.float16,
        # device_map=device_map,
    )

    # if training_args.local_rank == 0:
    #     print(model)
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(name)
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)
    train_dataset, visual_names = load_datasets(data_args)

    # Initialize our Trainer
    trainer = LLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval 
        and not is_torch_tpu_available() else None,
        )

    if training_args.do_eval:
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

        # metrics = trainer.evaluate()
        tokenizer = LlamaTokenizer.from_pretrained('trained_models/llama_tokenizer')
        model = trainer.get_model()
        image_dirs = ['None', 'None', 'None', 'None', 'None']
        video_dirs = ['None', 'None', 'None', 'data/avsd/frames/7UPGT', 'data/avsd/frames/3MSZA']
        audio_dirs = ['None', 'None', 'None', 'data/avsd/audios/7UPGT', 'data/avsd/audios/3MSZA']

        instructions = [prompt.format('Give three tips for staying healthy.'), 
        prompt.format('Tell me who is the president of USA?'), 
        prompt.format('Which season is hotter, winter or summer?'), 
        prompt.format('Does the woman eat or drink anything?'),
        prompt.format('What\'s on the table next to her?')]

        inference_generation(model, tokenizer, image_dirs, audio_dirs, video_dirs, instructions)

        # exit()
        # batch_size = 24
        # # dataset_name = 'avsd'
        # dataset_name = data_args.dataset_name
        # val_dir = 'data/{}/{}_val_inference.json'.format(dataset_name, dataset_name)
        # all_val_examples = json_load(val_dir)

        # image_dirs = [e['image'] for e in all_val_examples]
        # video_dirs = [e['video'] for e in all_val_examples]
        # audio_dirs = [e['audio'] for e in all_val_examples]
        # instructions = [prompt.format(e['instruction']) for e in all_val_examples]
        # responses = [e['response'] for e in all_val_examples]

        # batch_inference_generation(training_args, model, tokenizer, image_dirs, audio_dirs, video_dirs, instructions, responses, batch_size, dataset_name)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
