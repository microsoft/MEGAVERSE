# Introduction 
Code for MEGAVERSE. This codebase builds on the codebase of MEGA from [here](https://github.com/microsoft/Multilingual-Evaluation-of-Generative-AI-MEGA). MEGAVERSE covers new datasets (AfriQA, Belebele, X-RiSAWOZ, IN-22), and multimodal datasets (MarVL, XM-3600) along with new open-source models (Gemma, llama2, and Mistral).

![Datasets, Models and Modalities in MEGAVERSE](images/megaverse_tasks.png)

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

The framework requires keys and endpoints for [OpenAI API](https://platform.openai.com), and [HUggingFace API](https://huggingface.co/inference-api) for inferencing. Please place all the keys, endpoints and expected env variables under `envs/melange.env`

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

We also have shell scripts for all the datasets. The scripts reside in `scripts` for API based querying and `hf_scripts` for local model querying respectively.
To run multimodal benchmarks, refer to the README in `multimodal/README.md`


For contamination analysis, refer to the README in `contamination/README.md` file to run analysis for closed source models such as GPT-4 and PaLM2. For open-source contamination analysis, we referred to the work by Oren et. al [here](https://github.com/tatsu-lab/test_set_contamination).