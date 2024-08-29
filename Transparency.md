# Transparency Note

## Overview
MEGAVERSE benchmarks several Large Language Models on standard NLP benchmarks, covering 22 datasets and 83 languages including many low-resource languages.

## Objective
The objective of the MEGAVERSE project is gain an understanding of Large Language Model (LLM) and Small Language Model (SLM) performance on languages beyond English, by standardizing evaluation practices and covering a large number of languages.

## Audience
Model developers, researchers, application developers

## Key Features
MEGAVERSE is a benchmarking suite that covers the following 22 datasets - AfriQA, Belebele, X-RiSAWOZ, IN-22, MarVL, XM-3600, XNLI, IndicXNLI, GLUECoS NLI, PAWS-X, XCOPA, XStoryCloze, En-Es-CS, TyDiQA-GoldP, MLQA, XQUAD, IndicQA, PAN-X, UDPOS, Jigsaw, WinoMT. For a list of languages covered, please see the MEGAVERSE paper: https://aclanthology.org/2024.naacl-long.143.pdf

##Intended Uses
MEGAVERSE is a benchmarking suite for evaluating Large Language Model performance on non-English languages. Due to the limitations mentioned below, it should be supplemented with other forms of evaluation, such as human evaluation to gain a more comprehensive view on language model performance on target languages.

## Limitations
Model comparison - Access to commercial models (GPT, PaLM2, etc.) is via an API endpoint. These models might be running various post-processing modules and classifiers resulting in performance that is superior to Open models. 
Dataset contamination - it is possible that datasets present in MEGAVERSE are contaminated into pre-training or fine-tuning data of Large Language Models, so results should be used with caution. We perform the dataset contamination exercise on a few set of datasets for PaLM2 and GPT-4. We also perform a thorough analysis of the open-source models covered in MEGAVERSE. 
Prompt tuning -  LLMs are sensitive to prompting, and we do not perform extensive prompt tuning for the datasets in MEGAVERSE. We also do not experiment with prompting variations, such as translate-test and zero-shot cross-lingual prompting, or more complex strategies such as Chain of Thought prompting.
Focus on task accuracy - Although we include Responsible AI datasets, most of the datasets in MEGAVERSE measure task accuracy. Safety evaluation is an important aspect of evaluating LLMs on non-English languages that should be done before deployment.

## Best Practices for Performance
The prompts that we suggest in MEGAVERSE may not lead to optimal performance for all models, so we recommend tuning prompts using a dev set.

## Future Updates
Future updates to MEGAVERSE will be available on the Github repository of the project.

##Out of scope uses
MEGAVERSE is not intended to be used for any purposes other than research, benchmarking and evaluation
