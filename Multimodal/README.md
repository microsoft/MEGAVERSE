# MEGAVERSE Mutimodal 

## Introduction
This sub-module of the repository is meant to evaluate multimodal (vision-based) LLMs on two datasets.

| **Dataset** | **Task**           | **Paper** | **GitHub Repository** | **Hugging Face Dataset** |
|-------------|--------------------|-----------|-----------------------|--------------------------|
| MARVL       | Visual Reasoning   | [Paper](https://aclanthology.org/2021.emnlp-main.818/) | [link](https://github.com/marvl-challenge/marvl-code) | [link](https://huggingface.co/datasets/floschne/marvl) |
| XM3600      | Image Captioning   | [Paper](https://aclanthology.org/2022.emnlp-main.45/) | [link](https://google.github.io/crossmodal-3600/) | [link](https://huggingface.co/datasets/floschne/xm3600) |


Note that, for generalizability, the evaluation code expects the dataset in a HuggingFace dataset format, and you will need to preprocess the newer datasets for further evaluation, as shown in the examples above. 

## Requirements
- Same the parent repository.
- Add all the credentials (OpenAI, Google, HF) to a `.env` file to load them during evaluation.

## Models evaluated
- [GPT-4-Vision](https://platform.openai.com/docs/guides/vision)
- [Gemini-Pro Vision](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-pro-vision?hl=es-419&pli=1)
- LLaVA family ([BakLLaVA-v1](https://huggingface.co/llava-hf/bakLlava-v1-hf), [ViP-LLaVA](https://huggingface.co/llava-hf/vip-llava-13b-hf), [LLaVA-1.5](https://huggingface.co/llava-hf/llava-1.5-13b-hf))

## Evaluating new datasets
- Write a preprocessing script to convert your custom dataset into HF format, and push it to the hub (optional). It is strongly recommended to resize all images to $640 \times 480$ before passing them to the model, as that particular image size has shown to give the best performance, both in terms of quantitative scores, and processing time. 
- In the `utils.py` file, create a new function to load your dataset, and apply any additional preprocessing.
- Following `eval_marvl.py` or `eval_xm3600.py`, create a new evaluation script.

## Results
Please refer to our [paper](https://arxiv.org/abs/2311.07463) for a detailed analysis of the evaluation of these models on the aforementioned datasets.