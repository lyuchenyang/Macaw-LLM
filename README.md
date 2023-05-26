# M2LM: Multi-Modal Language Modeling with Image, Video, Audio, and Text Integration 🌐🖼️📹🎵📝
<div align="center">
  <img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License">
  <img src="https://img.shields.io/github/stars/lyuchenyang/M2LM" alt="Stars">
  <img src="https://img.shields.io/github/issues/lyuchenyang/M2LM" alt="Issues">
  <img src="https://img.shields.io/badge/python-3.8-blue.svg" alt="Python">
  
  
<!-- **Authors:** -->

Chenyang Lyu¹, Bingshuai Liu², Zefeng Du³, Longyue Wang⁴

<!-- **Affiliations:** -->

¹ Dublin City University, ² Xiamen University, ³ University of Macau, ⁴ Tencent AI Lab
</div>


M2LM is an exploratory endeavor that pioneers multi-modal language modeling by seamlessly combining image, video, audio, and text data, built upon the foundations of CLIP, Whisper, and LLaMA.

## Table of Contents 📚

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Alignment Strategy](#alignment-strategy)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work and Contributions](#future-work-and-contributions)

## Introduction 📖
<div align="center">
  <img src="alignment.png" alt="Figure Description or Alt Text" width="75%">
</div>

<!-- ![Figure Description or Alt Text](alignment.png) -->

In recent years, the field of language modeling has witnessed remarkable advancements. However, the integration of multiple modalities, such as images, videos, audios, and text, has remained a challenging task. M2LM is a model of its kind, bringing together state-of-the-art models for processing visual, auditory, and textual information, namely CLIP, Whisper, and LLaMA.

## Architecture 🔧

M2LM is composed of three main components:

1. **CLIP**: Responsible for encoding images and video frames.
2. **Whisper**: Responsible for encoding audio data.
3. **LLM**(LLaMA/Vicuna/Bloom): The language model that encodes instructions and generates responses.

The integration of these models allows M2LM to process and analyze multi-modal data effectively.

## Alignment Strategy 📏

Our novel alignment strategy enables faster adaptation by efficiently bridging multi-modal features to textual features. The process involves:

1. Encoding multi-modal features with CLIP and Whisper.
2. Feeding the encoded features into an attention function, wherein the multi-modal features serve as the query and the embedding matrix of LLaMA as the key and value.
3. Injecting the outputs into the input sequence (before instruction tokens) of LLaMA, allowing for a streamlined alignment process with minimal additional parameters.

## Installation 💻

To install M2LM, follow these steps:

1. Clone the repository: 
  `git clone https://github.com/lyuchenyang/M2LM.git`
2. Change to the M2LM directory: 
  `cd M2LM`
3. Install required packages: 
  `pip install -r requirements.txt`
4. Install ffmpeg: 
  `yum install ffmpeg -y`
5. Install apex: 
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
```

## Usage 🚀

1. Downloading dataset: 
	1) Text data: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json 
	2) Image data: https://cocodataset.org/#home 
	3) Video data: https://allenai.org/plato/charades/  and https://video-dialog.com/ 
2. Dataset preprocessing: 
	1) Place the data in three modalities to specific folders - 'data/text/', 'data/image/', 'data/video/' 
	2) Extract frames and audio from videos: `python preprocess_data.py` 
	3) Transform supervised data to dataset: `python preprocess_data_supervised.py`. 
	4) Transform unsupervised data to dataset: `python preprocess_data_unsupervised.py`
3. Training: ./train.sh - where you can specify the training parameters
4. Inference: ./inference.sh - where you can give any customized inputs

## Future Work and Contributions 🚀

While our model is still in its early stages, we believe that M2LM paves the way for future research in the realm of multi-modal language modeling. The integration of diverse data modalities holds immense potential for pushing the boundaries of artificial intelligence and enhancing our understanding of complex real-world scenarios. By introducing M2LM, we hope to inspire further exploration and innovation in this exciting area of study.

We welcome contributions from the community to improve and expand M2LM's capabilities. 🤝
