from tqdm import tqdm
import pickle
import json
import codecs
import requests
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import random

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


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

    
if __name__ == '__main__':
    sample_frames_from_video()
    extract_audio_from_video()
