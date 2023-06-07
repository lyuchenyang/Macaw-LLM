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

def preprocess_vqa2_to_val_dataset():
    all_examples = json_load('data/vqa/mscoco_val2014_annotations.json')['annotations']
    all_questions = json_load('data/vqa/OpenEnded_mscoco_val2014_questions.json')
    all_questions = {e['question_id']: [e['image_id'], e['question']] for e in all_questions['questions']}

    all_val_examples = []
    for ind, e in enumerate(tqdm(all_examples)):
        
        _image_dir = e['image_path']
        if len(_image_dir.split('_')[-1].split('.')[0]) < 12:
            i_str = _image_dir.split('_')[-1].split('.')[0]
            n_str = '0' * (12 - len(i_str)) + i_str
            _image_dir = _image_dir.replace(i_str, n_str)
        e = {
            'image': _image_dir,
            'video': 'None',
            'audio': 'None',
            'instruction': all_questions[e['question_id']][1],
            'response': e['multiple_choice_answer']
        }
        all_val_examples.append(e)
    
    data = {'data': all_val_examples}
    json_dump(data, 'data/vqa/vqa_val_inference.json')


def preprocess_avsd_to_val_dataset():
    import os
    metadata_dir = 'data/avsd/avsd_val.json'
    metadata = json_load(metadata_dir)

    all_val_examples = []
    path = 'data/avsd/'
    for ind, key in enumerate(tqdm(metadata)):
        md = metadata[key]

        video_dir = os.path.join(path, 'frames/{}'.format(key))
        audio_dir = os.path.join(path, 'audios/{}.mp4.wav'.format(key))

        all_t_q = []
        for dialog in md['data']:
            q = dialog['question'] + ' ' + dialog['answer']
            e = {
                'image': 'None',
                'video': video_dir,
                'audio': audio_dir,
                'instruction': dialog['question'],
                'response': dialog['answer']
            }
            all_val_examples.append(e)
    
    data = {'data': all_val_examples}

    json_dump(all_val_examples, 'data/avsd/avsd_val_inference.json')


def preprocess_vqa2_to_tensor_dataset(all_visual_names, tokenizer):
    all_examples = json_load('data/vqa/mscoco_train2014_annotations.json')['annotations']
    all_questions = json_load('data/vqa/OpenEnded_mscoco_train2014_questions.json')
    all_questions = {e['question_id']: [e['image_id'], e['question']] for e in all_questions['questions']}

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
            'instruction': all_questions[e['question_id']][1],
            'input': "",
            'output': e['multiple_choice_answer']
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
    # pickle.dump([all_images, all_null_audios, all_null_videos, all_texts, all_labels], open('data/vqa/vqa.cache', "wb"), protocol=4)
    tokenized_texts = tokenizer(all_textual_inputs, max_length=max_length, padding='max_length', truncation=True)
    tokenized_texts['labels'] = all_native_labels
    tokenized_texts['images'] = all_images
    tokenized_texts['audios'] = all_null_audios
    tokenized_texts['videos'] = all_null_videos

    # pickle.dump(tokenized_texts, open('data/vqa/vqa_new.cache', "wb"), protocol=4)

    video_names = {'data': all_image_names}
    # json_dump(video_names, 'data/vqa/vqa_video_names.json')

    return all_textual_inputs, all_native_labels, all_images, all_null_audios, all_null_videos


def preprocess_alpaca_to_tensor_dataset(tokenizer):
    all_examples = json_load('data/alpaca_data/alpaca_data.json')

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


def extract_audio_from_video():
    import moviepy.editor as mp

    path = './data/avsd/videos/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for f in tqdm(onlyfiles):
        dir = path + f
        clip = mp.VideoFileClip(dir)
        clip.audio.write_audiofile('./data/avsd/audios/{}.wav'.format(f))
    return


def sample_frames_from_video():
    # Importing all necessary libraries
    import cv2

    path = 'data/avsd/videos/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    frames_per_video = 120
    for f in tqdm(onlyfiles):
        # Read the video from specified path
        cam = cv2.VideoCapture(path + f)

        # frame
        currentframe = 0
        all_frames = []
        while (True):

            # reading from frame
            ret, frame = cam.read()

            if ret:
                all_frames.append(frame)
                currentframe += 1
            else:
                break
        lens = len(all_frames)
        if lens >= frames_per_video:
            interval = lens // frames_per_video

            frame_ind = [i * interval for i in range(frames_per_video)]
            for i in range(len(frame_ind)):
                if frame_ind[i] >= lens:
                    frame_ind[i] = lens - 1
            frame_ind[-1] = lens - 1
            sampled_frames = [all_frames[i] for i in frame_ind]
        else:
            sampled_frames = sorted(draw_samples([i for i in range(len(all_frames))], frames_per_video))
            sampled_frames = [all_frames[i] for i in sampled_frames]

        for ind, frame in enumerate(sampled_frames):
            cv2.imwrite('./data/avsd/frames/{}_{}.jpg'.format(f, str(ind)), frame)

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()


def preprocess_avsd_to_tensor_dataset(all_visual_names, tokenizer):
    image_dir = 'data/avsd/frames/'
    audio_dir = 'data/avsd/audios/'

    train_metadata_dir = 'data/avsd/avsd_train.json'
    val_metadata_dir = 'data/avsd/avsd_val.json'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.random.manual_seed(0)
    max_length = 256
    def read_image_and_audio(metadata_dir, split='train'):
        metadata = json_load(metadata_dir)

        all_video_names = []
        all_videos, all_audios, all_texts, all_null_images = [], [], [], []
        all_labels = []

        all_textual_inputs = []
        all_native_labels = []
        for ind, key in enumerate(tqdm(metadata)):
            md = metadata[key]
            all_video_names.append(key)

            all_frames = ind

            all_t_q = []
            for dialog in md['data']:
                q = dialog['question'] + ' ' + dialog['answer']
                prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n {} \n\n"
                q = prompt.format(dialog['question'], dialog['answer'])
                t_all = tokenizer.encode(q, max_length=max_length)

                q_input = q.split(' Response:')[0] + ' Response:'

                if len(t_all) > max_length:
                    t_all = t_all[:max_length]
                else:
                    t_all = t_all + [tokenizer.pad_token_id] * (max_length - len(t_all))

                len_t_q = len(tokenizer.encode(q_input)) - 1
                labels = [IGNORE_INDEX] * len_t_q + t_all[len_t_q:]
                if len(labels) > max_length:
                    labels = labels[:max_length]
                if len(labels) < max_length:
                    labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
                all_textual_inputs.append(q)
                all_native_labels.append(labels)

                all_videos.append(all_visual_names[key])
                all_audios.append(all_visual_names[key])
                all_null_images.append(-1)
                all_texts.append(torch.tensor([t_all], dtype=torch.int))
                all_labels.append(torch.tensor([labels], dtype=torch.int))

        video_names = {'split': split, 'data': all_video_names}

        tokenized_texts = tokenizer(all_textual_inputs, max_length=max_length, padding='max_length', truncation=True)
        tokenized_texts['labels'] = all_native_labels
        tokenized_texts['images'] = all_null_images
        tokenized_texts['audios'] = all_audios
        tokenized_texts['videos'] = all_videos
        
        return all_textual_inputs, all_native_labels, all_null_images, all_audios, all_videos

    all_textual_inputs, all_native_labels, all_images, all_audios, all_videos = read_image_and_audio(train_metadata_dir, split='train')

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
    all_visual_names = json_load('data/all_visual_names.json')['dict']
    tokenizer = AutoTokenizer.from_pretrained('trained_models/llama_tokenizer')

    # Chenyang: 2023-05-21, add special tokens

    special_tokens_dict = {'additional_special_tokens': ['<image>', '</image>', '<audio>', '</audio>', '<video>', '</video>']}

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    tokenizer.padding_side = "right"

    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    tokenizer.save_pretrained('trained_models/llama_tokenizer')

    all_image_data = preprocess_vqa2_to_tensor_dataset(all_visual_names, tokenizer)
    all_tetx_data = preprocess_alpaca_to_tensor_dataset(tokenizer)
    all_video_data = preprocess_avsd_to_tensor_dataset(all_visual_names, tokenizer)

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
    pickle.dump(tokenized_texts, open('data/train_total_new_vname.cache', "wb"), protocol=4)


def combine_visual_and_audio_names():

    all_image_names = []

    def add_image_names(dir=None):
        all_examples = json_load(dir)['annotations']

        for ind, e in enumerate(tqdm(all_examples)):
            
            _image_dir = e['image_path']
            if len(_image_dir.split('_')[-1].split('.')[0]) < 12:
                i_str = _image_dir.split('_')[-1].split('.')[0]
                n_str = '0' * (12 - len(i_str)) + i_str
                _image_dir = _image_dir.replace(i_str, n_str)

            all_image_names.append(_image_dir)
    add_image_names('data/vqa/mscoco_train2014_annotations.json')
    add_image_names('data/vqa/mscoco_val2014_annotations.json')

    all_video_names = []

    train_metadata_dir = 'data/avsd/avsd_train.json'
    val_metadata_dir = 'data/avsd/avsd_val.json'

    def add_video_names(metadata_dir):
        metadata = json_load(metadata_dir)
        for ind, key in enumerate(tqdm(metadata)):
            all_video_names.append(key)

    add_video_names(train_metadata_dir)
    add_video_names(val_metadata_dir)
    all_names = all_image_names + all_video_names

    all_names_dict = {k:ind for ind, k in enumerate(all_names)}
    all_names = {'dict': all_names_dict, 'list': all_names}

    json_dump(all_names, 'data/all_visual_names.json')
    
if __name__ == '__main__':
    combine_visual_and_audio_names()
    preprocess_all_datasets()
