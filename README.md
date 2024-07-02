# Introduction 
Code for MEGAVERSE. This codebase builds on the codebase of MEGA from [here](https://github.com/microsoft/Multilingual-Evaluation-of-Generative-AI-MEGA)

# Getting Started

### Dependencies
- Compatible with Python3.8
- The necessary packages can be install through requirements.txt.

### Setup

We recommend creating a virtual environment for the framework (optional):

```shell
$ [sudo] pip install virtualenv
$ virtualenv -p python3.8 megaenv
$ source megaenv/bin/activate
```

Install the required packages by running:


```shell
pip install -r requirements.txt
```

The framework requires keys and endpoints for [OpenAI API](https://platform.openai.com), [Azure Translation API](https://www.microsoft.com/en-us/translator/business/translator-api/) and [HUggingFace API](https://huggingface.co/inference-api) for inferencing. Please place all the keys, endpoints and expected env variables under `envs/melange.env`

#### Expected env variables
1. `OPENAI_END_POINT`
2. `OPENAI_API_KEY`
3. `OPENAI_API_TYPE`
4. `OPENAI_API_VERSION`
5. `HF_API_URL`
6. `HF_API_KEY`

# Running Evaluation

## XNLI

The following language prompts are currently available in the framework

Language | Available Prompts
-------- | -----------------
en (English) | GPT-3 Style, MNLI crowdsource, always/sometimes/never, based on the previous passage,...
hi (Hindi) | English GPT-3 Style, Handcrafted GPT-3 Style, English based on the previous passage, Handcrafted based on the previous passage, ...

The full list can be found in `promptsource`. Check [`promptsource/README.md`](promptsource/README.md) for details.

To run the evaluation on XNLI, execute the following command
```shell
$ python -m mega.eval_xnli \
    -t {Target Language} \
    -p {Pivot Language} \
    -k {Number of Few-shot Examples} \
    --tgt_prompt_name {Prompt Name For Target} \ 
    --pivot_prompt_name {Prompt Name For Pivot} \
    --model {GPT Model to Evaluate} \
    {--translate-test}
```

An example command would look like:

```shell
python -m mega.eval_xnli \
    -p hi \
    -t hi \
    -k 8 \
    --pivot_prompt_name "Handcrafted based on the previous passage" \
    --tgt_prompt_name "Handcrafted based on the previous passage" \
    --model gpt-35-turbo
```

**Other tasks to be added soon!**

## Extending the framework to workf for other Classification/QA tasks:

Extending to other classification and QA tasks is simple. First create the prompts for different languages for a selected task using promptsource (next section). Then we only need to load the dataset for the task and the prompt templates to perform evaluation. Check sample code below for PAWS-X:
```python
# Import the necessary modules to run evaluation
from mega.eval.eval_cls import evaluate_model
from mega.data.data_utils import choose_few_shot_examples

# Import datasets and promptsource libraries
from datasets import load_dataset
from promptsource.templates import DatasetTemplates


# Load dataset of your choice
dataset = "paws-x"
src_lang = "en" #Can change the language from en to the language of your choice 
tgt_lang = "en" #Similarly language here can be changed, if it is same as src_lang then monolingual, else zero-shot
train_dataset = load_dataset(dataset, src_lang)["train"] 
test_dataset = load_dataset(dataset, tgt_lang)["test"]

# Load prompt templates for the dataset
prompt_name = "Meaning" # Name of the prompt created by you on promptsource
train_prompt = DatasetTemplates(f"{dataset}/{src_lang}")[prompt_name]
test_prompt = DatasetTemplates(f"{dataset}/{tgt_lang}")[prompt_name]

# Run evaluation
accuracy = evaluate_model(
        train_dataset,
        test_dataset,
        train_prompt,
        test_prompt,
        model="gpt-35-turbo", #Can change this to BLOOM also
        few_shot_size=4, #Number of few-shot examples
        save_preds_path="results/preds.csv",#Any path where you would like to store predictions,
        temperature=0.1, # Temperature parameter for GPT-3x generations
    )
print(accuracy)
```


# Creating New Prompts
Adapted from [`promptsource/README.md`](promptsource/README.md)

PromptSource provides a Web-based GUI that enables developers to write prompts in a templating language and immediately view their outputs on different examples.

There are 3 modes in the app:
- **Sourcing**: create and write new prompts
- **Prompted dataset viewer**: check the prompts you wrote (or the existing ones) on the entire dataset
- **Helicopter view**: aggregate high-level metrics on the current state of P3

![ALT](promptsource/assets/promptsource_app.png)

To launch the app locally, please first make sure you have followed the steps in [Setup](#setup), and from the root directory of the repo, run:
```bash
cd promptsource
streamlit run promptsource/app.py
```

You can also browse through existing prompts on the [hosted version of PromptSource](https://bigscience.huggingface.co/promptsource). Note the hosted version disables the Sourcing mode (`streamlit run promptsource/app.py -- --read-only`).

### Writing prompts
Before creating new prompts, you should read the [contribution guidelines](CONTRIBUTING.md) which give an step-by-step description of how to contribute to the collection of prompts.
