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

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
DEFAULT_PAD_TOKEN = "[PAD]"

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


def preprocess_coco_to_tensor_dataset(all_visual_names):
    all_examples = json_load('data/generated_examples_coco.json')
    tokenizer = LlamaTokenizer.from_pretrained('trained_models/llama_tokenizer')

    max_length = 256
    all_image_names = []
    all_images, all_null_audios, all_null_videos = [], [], []
    all_texts, all_labels = [], []
    random_indices = draw_samples([i for i in range(len(all_examples))], 60000)
    random_indices = {i: i for i in random_indices}
    index = 0

    all_textual_inputs = []
    all_native_labels = []
    for ind, e in enumerate(tqdm(all_examples)):
        if ind not in random_indices:
            continue
        all_image_names.append(e['image_path'])
        e = {
            'instruction': e['instruction'],
            'input': "",
            'output': e['response']
        }
        texts = PROMPT_DICT['prompt_input'].format(e['instruction'], e['input']) if e['input'] != "" else PROMPT_DICT['prompt_no_input'].format(e['instruction'])
        full_texts = texts + '\n {} \n\n'.format(e['output'])

        all_textual_inputs.append(full_texts)
        t_all = tokenizer.encode(full_texts)
        
        _image_dir = all_image_names[-1]
        if len(_image_dir.split('_')[-1].split('.')[0]) < 12:
            i_str = _image_dir.split('_')[-1].split('.')[0]
            n_str = '0' * (12 - len(i_str)) + i_str
            _image_dir = _image_dir.replace(i_str, n_str)

        all_images.append(all_visual_names[_image_dir])
        index += 1
        
        t_texts = tokenizer.encode(texts)


        if len(t_texts) >= max_length:
            continue
        if len(t_all) > max_length:
            t_all = t_all[:max_length]
        if len(t_all) < max_length:
            t_all = t_all + [32006] * (max_length - len(t_all))

        prefix_len = len(t_texts) - 1
        labels = [-100] * prefix_len + t_all[prefix_len:]
        if len(labels) > max_length:
            labels = labels[:max_length]
        if len(labels) < max_length:
            labels = labels + [-100] * (max_length - len(labels))
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

    video_names = {'data': all_image_names}

    return all_textual_inputs, all_native_labels, all_images, all_null_audios, all_null_videos


def preprocess_alpaca_to_tensor_dataset():
    all_examples = json_load('data/alpaca_data/alpaca_data.json')

    tokenizer = LlamaTokenizer.from_pretrained('trained_models/llama_tokenizer')

    max_length = 256
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
            t_all = t_all + [32006] * (max_length - len(t_all))
        all_textual_inputs.append(full_texts)

        prefix_len = len(t_texts) - 1
        labels = [-100] * prefix_len + t_all[prefix_len:]        
        if len(labels) > max_length:
            labels = labels[:max_length]
        if len(labels) < max_length:
            labels = labels + [-100] * (max_length - len(labels))    
        
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


# xxx: 2023-03-21
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def preprocess_avsd_to_tensor_dataset(all_visual_names):
    import clip
    import torch
    from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel, LlamaForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained('trained_models/llama_tokenizer')

    train_metadata_dir = 'data/generated_examples_avsd.json'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.random.manual_seed(0)
    max_length = 256
    def read_image_and_audio(metadata_dir):
        metadata = json_load(metadata_dir)

        all_videos, all_audios, all_texts, all_null_images = [], [], [], []
        all_labels = []

        all_textual_inputs = []
        all_native_labels = []
        for ind, e in enumerate(tqdm(metadata)):

            prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n {} \n\n"
            q = prompt.format(e['instruction'], e['response'])
            t_all = tokenizer.encode(q, max_length=max_length)

            q_input = q.split(' Response:')[0] + ' Response:'

            if len(t_all) > max_length:
                t_all = t_all[:max_length]
            else:
                t_all = t_all + [32006] * (max_length - len(t_all))

            len_t_q = len(tokenizer.encode(q_input)) - 1
            labels = [-100] * len_t_q + t_all[len_t_q:]
            if len(labels) > max_length:
                labels = labels[:max_length]
            if len(labels) < max_length:
                labels = labels + [-100] * (max_length - len(labels))
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


def resize_images():
    from PIL import Image

    path = 'data/avsd/videos/frames/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # onlyfiles = set([f.split('_')[0] for f in onlyfiles])

    for f in tqdm(onlyfiles):
        ind = int(f.replace('.jpg', '').split('_')[1])
        image = Image.open(path + f)
        image.thumbnail((336, 336))
        image.save(path.replace('frames', 'frames_resize') + f)
    # print(t)


def preprocess_all_datasets():
    all_visual_names = json_load('data/all_visual_names_instruction.json')['dict']
    tokenizer = LlamaTokenizer.from_pretrained('trained_models/llama_tokenizer')

    all_image_data = preprocess_vqa2_to_tensor_dataset(all_visual_names)
    all_tetx_data = preprocess_alpaca_to_tensor_dataset()
    all_video_data = preprocess_avsd_to_tensor_dataset(all_visual_names)

    def draw_examples(lis, num):
        ri = draw_samples([i for i in range(len(lis))], num)
        return ri

    ra, rb, rc = None, None, None

    all_dataset = []
    i = 0
    for a,b,c in zip(all_image_data, all_tetx_data, all_video_data):
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

    max_length = 256
    tokenized_texts = tokenizer(all_dataset[0], max_length=max_length, padding='max_length', truncation=True)
    tokenized_texts['labels'] = all_dataset[1]
    
    tokenized_texts['images'] = all_dataset[2]
    tokenized_texts['audios'] = all_dataset[3]
    tokenized_texts['videos'] = all_dataset[4]

    for k in tokenized_texts:
        print(k)

    # import ipdb
    # ipdb.set_trace()
    pickle.dump(tokenized_texts, open('data/train_total_new_instruction.cache', "wb"), protocol=4)


def combine_visual_and_audio_names():

    all_names = []

    image_examples = json_load('data/generated_examples_coco.json')
    video_examples = json_load('data/generated_examples_avsd.json')

    for e in image_examples:
        all_names.append(e['id'])
    
    for e in video_examples:
        all_names.append(e['id'])
    
    all_names_dict = {k:ind for ind, k in enumerate(all_names)}
    all_names = {'dict': all_names_dict, 'list': all_names}

    json_dump(all_names, 'data/all_visual_names_instruction.json')
    
if __name__ == '__main__':
    combine_visual_and_audio_names()
    preprocess_all_datasets()
