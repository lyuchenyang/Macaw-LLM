from tqdm import tqdm
import pickle
import json
import codecs
import requests
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer, LlamaTokenizer
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import random
import clip
import torch
from transformers import AutoFeatureExtractor, AutoModel, LlamaForCausalLM

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

# xxx: 2023-03-21
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:"
    ),
}


def preprocess_coco_to_tensor_dataset(data_args, all_visual_names, tokenizer):
    train_metadata_dir = 'data/generated_examples_coco.json' if data_args.image_instruction_file is None else data_args.image_instruction_file
    all_examples = json_load(train_metadata_dir)['data']

    max_length = data_args.max_seq_len
    all_images, all_null_audios, all_null_videos = [], [], []
    all_texts, all_labels = [], []

    all_textual_inputs = []
    all_native_labels = []
    for ind, e in enumerate(tqdm(all_examples)):
        if 'caption' in e['instruction'] or 'caption' in e['response'] or ' no ' in e['response'] or 'not' in e['response']:
            continue
        all_images.append(all_visual_names[e['id']])
        e = {
            'instruction': e['instruction'],
            'input': "",
            'output': e['response']
        }
        texts = PROMPT_DICT['prompt_input'].format(e['instruction'], e['input']) if e['input'] != "" else PROMPT_DICT['prompt_no_input'].format(e['instruction'])
        full_texts = texts + '\n {} \n\n'.format(e['output'])

        all_textual_inputs.append(full_texts)
        t_all = tokenizer.encode(full_texts)
        
        t_texts = tokenizer.encode(texts)

        if len(t_texts) >= max_length:
            continue
        if len(t_all) > max_length:
            t_all = t_all[:max_length]
        if len(t_all) < max_length:
            t_all = t_all + [tokenizer.pad_token_id] * (max_length - len(t_all))

        prefix_len = len(t_texts) - 1
        labels = [IGNORE_INDEX] * prefix_len + t_all[prefix_len:]
        if len(labels) > max_length:
            labels = labels[:max_length]
        if len(labels) < max_length:
            labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
        all_texts.append(torch.tensor([t_all], dtype=torch.int))
        all_labels.append(torch.tensor([labels], dtype=torch.int))
        all_native_labels.append(labels)

    all_null_audios = [-1] * len(all_images)
    all_null_videos = all_null_audios
    tokenized_texts = tokenizer(all_textual_inputs, max_length=max_length, padding='max_length', truncation=True)
    tokenized_texts['labels'] = all_native_labels
    tokenized_texts['images'] = all_images
    tokenized_texts['audios'] = all_null_audios
    tokenized_texts['videos'] = all_null_videos

    return all_textual_inputs, all_native_labels, all_images, all_null_audios, all_null_videos


def preprocess_alpaca_to_tensor_dataset(data_args, tokenizer):
    train_metadata_dir = 'data/alpaca_data/alpaca_data.json' if data_args.text_instruction_file is None else data_args.text_instruction_file

    all_examples = json_load(train_metadata_dir)

    max_length = data_args.max_seq_len
    all_null_images, all_null_audios, all_null_videos = [], [], []
    all_texts, all_labels = [], []

    all_textual_inputs = []
    all_native_labels = []
    for ind, e in enumerate(tqdm(all_examples)):
        texts = PROMPT_DICT['prompt_input'].format(e['instruction'], e['input']) if e['input'] != "" else PROMPT_DICT['prompt_no_input'].format(e['instruction'])
        full_texts = texts + '\n {} \n\n'.format(e['output'])
        t_all = tokenizer.encode(full_texts)

        t_texts = tokenizer.encode(texts)
        if len(t_texts) >= max_length:
            continue
        if len(t_all) > max_length:
            t_all = t_all[:max_length]
        if len(t_all) < max_length:
            t_all = t_all + [tokenizer.pad_token_id] * (max_length - len(t_all))
        all_textual_inputs.append(full_texts)

        prefix_len = len(t_texts) - 1
        labels = [IGNORE_INDEX] * prefix_len + t_all[prefix_len:]        
        if len(labels) > max_length:
            labels = labels[:max_length]
        if len(labels) < max_length:
            labels = labels + [IGNORE_INDEX] * (max_length - len(labels))    
        
        all_texts.append(t_all)
        all_labels.append(labels)
        all_native_labels.append(labels)

    all_null_images = [-1] * len(all_texts)
    all_null_audios = all_null_images
    all_null_videos = all_null_images 

    tokenized_texts = tokenizer(all_textual_inputs, max_length=max_length, padding='max_length', truncation=True)
    tokenized_texts['labels'] = all_native_labels
    tokenized_texts['images'] = all_null_images
    tokenized_texts['audios'] = all_null_audios
    tokenized_texts['videos'] = all_null_videos

    return all_textual_inputs, all_native_labels, all_null_images, all_null_audios, all_null_videos


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def preprocess_avsd_to_tensor_dataset(data_args, all_visual_names, tokenizer):
    train_metadata_dir = 'data/generated_examples_avsd.json' if data_args.video_instruction_file is None else data_args.video_instruction_file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.random.manual_seed(0)
    max_length = data_args.max_seq_len
    def read_image_and_audio(metadata_dir):
        metadata = json_load(metadata_dir)['data']

        all_videos, all_audios, all_texts, all_null_images = [], [], [], []
        all_labels = []

        all_textual_inputs = []
        all_native_labels = []
        for ind, e in enumerate(tqdm(metadata)):
            if 'caption' in e['instruction'] or 'caption' in e['response'] or ' no ' in e['response'] or 'not' in e['response']:
                continue

            prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n {} \n\n"
            q = prompt.format(e['instruction'], e['response'])
            t_all = tokenizer.encode(q, max_length=max_length, truncation=True)

            q_input = q.split(' Response:')[0] + ' Response:'

            if len(t_all) > max_length:
                t_all = t_all[:max_length]
            if len(t_all) < max_length:
                t_all = t_all + [tokenizer.pad_token_id] * (max_length - len(t_all))

            len_t_q = len(tokenizer.encode(q_input)) - 1
            labels = [IGNORE_INDEX] * len_t_q + t_all[len_t_q:]
            if len(labels) > max_length:
                labels = labels[:max_length]
            if len(labels) < max_length:
                labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
            all_textual_inputs.append(q)
            all_native_labels.append(labels)

            all_videos.append(all_visual_names[e['id']])
            all_audios.append(all_visual_names[e['id']])
            all_null_images.append(-1)
            all_texts.append(torch.tensor([t_all], dtype=torch.int))
            all_labels.append(torch.tensor([labels], dtype=torch.int))

        tokenized_texts = tokenizer(all_textual_inputs, max_length=max_length, padding='max_length', truncation=True)
        tokenized_texts['labels'] = all_native_labels
        tokenized_texts['images'] = all_null_images
        tokenized_texts['audios'] = all_audios
        tokenized_texts['videos'] = all_videos
        
        return all_textual_inputs, all_native_labels, all_null_images, all_audios, all_videos

    all_textual_inputs, all_native_labels, all_images, all_audios, all_videos = read_image_and_audio(train_metadata_dir)

    return all_textual_inputs, all_native_labels, all_images, all_audios, all_videos


def preprocess_all_datasets(model_args, data_args, all_visual_names):
    # all_visual_names = json_load(data_args.visual_names_file)['dict']
    tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name)

    # Chenyang: 2023-05-21, add special tokens

    # special_tokens_dict = {'additional_special_tokens': ['<image>', '</image>', '<audio>', '</audio>', '<video>', '</video>']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # if tokenizer.pad_token is None or tokenizer.pad_token != DEFAULT_PAD_TOKEN:
    #     tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    # tokenizer.padding_side = "right"

    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "additional_special_tokens": ['<image>', '</image>', '<audio>', '</audio>', '<video>', '</video>']
        }
    )
    tokenizer.padding_side = "right"


    tokenizer.save_pretrained(model_args.tokenizer_name)

    all_image_data = preprocess_coco_to_tensor_dataset(data_args, all_visual_names, tokenizer)
    all_text_data = preprocess_alpaca_to_tensor_dataset(data_args, tokenizer)
    all_video_data = preprocess_avsd_to_tensor_dataset(data_args, all_visual_names, tokenizer)

    def draw_examples(lis, num):
        ri = draw_samples([i for i in range(len(lis))], num)
        return ri

    ra, rb, rc = None, None, None

    all_dataset = []
    i = 0
    for a,b,c in zip(all_image_data, all_text_data, all_video_data):
        if ra == None:
            print(len(a), len(b), len(c))
            ra = draw_examples(a, 50000)
            rb = draw_examples(b, 50000)
            rc = draw_examples(c, 50000)

            a = [a[i] for i in ra]
            b = [b[i] for i in rb]

            c = [c[i] for i in rc]

            new_lis = a + b + c
            print(len(new_lis))
            all_dataset.append(new_lis)
        else:
            print(len(a), len(b), len(c))
            a = [a[i] for i in ra]
            b = [b[i] for i in rb]

            c = [c[i] for i in rc]
            new_lis = a + b + c
            print(len(new_lis))
            all_dataset.append(new_lis)
        i += 1

    max_length = data_args.max_seq_len
    tokenized_texts = tokenizer(all_dataset[0], max_length=max_length, padding='max_length', truncation=True)
    tokenized_texts['labels'] = all_dataset[1]
    
    tokenized_texts['images'] = all_dataset[2]
    tokenized_texts['audios'] = all_dataset[3]
    tokenized_texts['videos'] = all_dataset[4]

    for k in tokenized_texts:
        print(k)

    # import ipdb
    # ipdb.set_trace()
    # pickle.dump(tokenized_texts, open('data/train_total_new_instruction_1.cache', "wb"), protocol=4)

    pickle.dump(tokenized_texts, open(data_args.train_file, "wb"), protocol=4)
    return tokenized_texts


def combine_visual_and_audio_names(data_args, image_data_dir='data/generated_examples_coco.json', video_data_dir='data/generated_examples_avsd.json'):

    all_names = {}

    image_examples = json_load(image_data_dir)['data']
    video_examples = json_load(video_data_dir)['data']

    for e in image_examples:
        if e['id'] not in all_names:
            all_names[e['id']] = len(all_names)
    
    for e in video_examples:
        if e['id'] not in all_names:
            all_names[e['id']] = len(all_names)    

    all_names_list = [k for k in all_names]
    all_names = {k: ind for ind, k in enumerate(all_names_list)}
    all_names = {'dict': all_names, 'list': all_names_list}

    # if not os.path.exists(data_args.visual_names_file):
    json_dump(all_names, data_args.visual_names_file)
    return all_names

def tokenize_all_datasets(model_args, data_args):
    # all_visual_names = None
    all_visual_names = combine_visual_and_audio_names(data_args, data_args.image_instruction_file, data_args.video_instruction_file)
    tokenized_data = preprocess_all_datasets(model_args, data_args, all_visual_names['dict'])
    return tokenized_data, all_visual_names

if __name__ == '__main__':
    combine_visual_and_audio_names()
    preprocess_all_datasets()
    pass