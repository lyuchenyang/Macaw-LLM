<div align="center">
  <img src="https://raw.githubusercontent.com/lyuchenyang/Macaw-LLM/main/assets/logo-text.png" alt="Logo" width="200">
</div>

# Macaw-LLM: Multi-Modal Language Modeling with Image, Video, Audio, and Text Integration 🌐🖼️📹🎵📝
<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version">
<img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/lyuchenyang/Macaw-LLM?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/lyuchenyang/Macaw-LLM?color=red" alt="Issues">
<img src="https://img.shields.io/badge/python-3.8-purple.svg" alt="Python">
  
  
<!-- **Authors:** -->

**_¹ [Chenyang Lyu](https://lyuchenyang.github.io), ² Bingshuai Liu, ³ [Minghao Wu](https://minghao-wu.github.io/), ⁴ [Zefeng Du](https://seeledu.github.io/index-en.html),_**

**_⁵ [Xinting Huang](https://timhuang1.github.io/), ⁵ [Zhaopeng Tu](http://www.zptu.net/), ⁵ [Shuming Shi](https://shumingshi.github.io/), ⁵ [Longyue Wang](http://www.longyuewang.com/)_**


<!-- **Affiliations:** -->

_¹ Dublin City University, ² Xiamen University, ³ Monash University, ⁴ University of Macau, ⁵ Tencent AI Lab_
</div>


Macaw-LLM is an exploratory endeavor that pioneers multi-modal language modeling by seamlessly combining image, video, audio, and text data, built upon the foundations of CLIP, Whisper, and LLaMA.

## Table of Contents 📚

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Alignment Strategy](#alignment-strategy)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work and Contributions](#future-work-and-contributions)

## Introduction 📖
<div align="center">
  <img src="assets/alignment.png" alt="Figure Description or Alt Text" width="70%">
</div>

<!-- ![Figure Description or Alt Text](alignment.png) -->

In recent years, the field of language modeling has witnessed remarkable advancements. However, the integration of multiple modalities, such as images, videos, audios, and text, has remained a challenging task. Macaw-LLM is a model of its kind, bringing together state-of-the-art models for processing visual, auditory, and textual information, namely CLIP, Whisper, and LLaMA.

## Key Features 🔑

Macaw-LLM boasts the following unique features:

1. **Simple & Fast Alignment**: Macaw-LLM enables seamless integration of multi-modal data through simple and fast alignment to LLM embeddings. This efficient process ensures quick adaptation of diverse data types.
2. **One-Stage Instruction Fine-Tuning**: Our model streamlines the adaptation process through one-stage instruction fine-tuning, promoting a more efficient learning experience.


## Architecture 🔧

Macaw-LLM is composed of three main components:

1. **CLIP**: Responsible for encoding images and video frames.
2. **Whisper**: Responsible for encoding audio data.
3. **LLM**(LLaMA/Vicuna/Bloom): The language model that encodes instructions and generates responses.

The integration of these models allows Macaw-LLM to process and analyze multi-modal data effectively.

## Alignment Strategy 📏

Our novel alignment strategy enables faster adaptation by efficiently bridging multi-modal features to textual features. The process involves:

1. Encoding multi-modal features with CLIP and Whisper.
2. Feeding the encoded features into an attention function, wherein the multi-modal features serve as the query and the embedding matrix of LLaMA as the key and value.
3. Injecting the outputs into the input sequence (before instruction tokens) of LLaMA, allowing for a streamlined alignment process with minimal additional parameters.

## Installation 💻

To install Macaw-LLM, follow these steps:

```bash
# Clone the repository
git clone https://github.com/lyuchenyang/Macaw-LLM.git

# Change to the Macaw-LLM directory
cd Macaw-LLM

# Install required packages
pip install -r requirements.txt

# Install ffmpeg
yum install ffmpeg -y

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..
```

## Usage 🚀

1. **Downloading dataset:** 
   - Text data: [stanford_alpaca/alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) 
   - Image data: [COCO Dataset](https://cocodataset.org/#home) 
   - Video data: [Charades](https://allenai.org/plato/charades/) and [Video Dialog](https://video-dialog.com/) 

2. **Dataset preprocessing:** 
   - Place the data in three modalities to specific folders - `data/text/`, `data/image/`, `data/video/`
   - Extract frames and audio from videos: 
     ```
     python preprocess_data.py
     ```
   - Transform supervised data to dataset: 
     ```
     python preprocess_data_supervised.py
     ```
   - Transform unsupervised data to dataset: 
     ```
     python preprocess_data_unsupervised.py
     ```

3. **Training:** 
   - Execute the training script (you can specify the training parameters inside):
     ```
     ./train.sh
     ```

4. **Inference:** 
   - Execute the inference script (you can give any customized inputs inside):
     ```
     ./inference.sh
     ```


## Future Work and Contributions 🚀

While our model is still in its early stages, we believe that Macaw-LLM paves the way for future research in the realm of multi-modal language modeling. The integration of diverse data modalities holds immense potential for pushing the boundaries of artificial intelligence and enhancing our understanding of complex real-world scenarios. By introducing Macaw-LLM, we hope to inspire further exploration and innovation in this exciting area of study.

We welcome contributions from the community to improve and expand Macaw-LLM's capabilities. 🤝

## ToDo 👨‍💻

- [ ] **More Language Models:** We aim to extend Macaw-LLM by incorporating additional language models like Dolly, BLOOM, T-5, etc. This will enable more robust and versatile processing and understanding of multi-modal data.

- [ ] **Multilingual Support:** Our next step is to support multiple languages, moving towards true multi-modal and multilingual language models. We believe this will significantly broaden Macaw-LLM's applicability and enhance its understanding of diverse, global contexts.


## Citation

```bibtex
@misc{Macaw-LLM,
  author = {Chenyang Lyu and Bingshuai Liu and Minghao Wu and Zefeng Du and Xinting Huang and Zhaopeng Tu and Shuming Shi and Longyue Wang},
  title = {Macaw-LLM: Multi-Modal Language Modeling with Image, Video, Audio, and Text Integration},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lyuchenyang/Macaw-LLM}},
}
```
