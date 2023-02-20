import requests
import openai
from typing import List, Dict, Union
from promptsource.templates import Template
from mega.prompting.prompting_utils import construct_prompt

openai.api_base = "https://gpttesting1.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2022-12-01"  # this may change in the future
HF_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"

with open("keys/openai_key.txt") as f:
    openai.api_key = f.read().split("\n")[0]

with open("keys/hf_key.txt") as f:
    HF_API_TOKEN = f.read().split("\n")[0]

SUPPORTED_MODELS = ["DaVinci003", "BLOOM"]


def gpt3x_completion(prompt: str, model: str, **model_params) -> str:

    """Runs the prompt over the GPT3.x model for text completion

    Args:
        - prompt (str) : Prompt String to be completed by the model
        - model (str) : GPT-3x model to use

    Returns:
        str: generated string
    """

    # Hit the api repeatedly till response is obtained
    while True:
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=model_params.get("max_tokens", 10),
                temperature=model_params.get("temperature", 1),
                top_p=model_params.get("top_p", 1),
            )
            break
        except (openai.error.APIConnectionError, openai.error.RateLimitError) as e:
            continue

    return response["choices"][0]["text"].strip()


def bloom_completion(prompt: str, **model_params) -> str:
    """Runs the prompt over BLOOM model for text completion

    Args:
        prompt (str): Prompt String to be completed by the model

    Returns:
        str: generated string
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def query(payload):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        return response.json()

    model_output = query(prompt)
    try:
        return model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
    except:
        import pdb

        pdb.set_trace()


def model_completion(prompt: str, model: str, **model_params) -> str:

    """Runs the prompt over one of the `SUPPORTED_MODELS` for text completion

    Args:
        - prompt (str) : Prompt String to be completed by the model
        - model (str) : Model to use

    Returns:
        str: generated string
    """

    if model == "DaVinci003":
        return gpt3x_completion(prompt, model, **model_params)

    if model == "BLOOM":
        return bloom_completion(prompt, **model_params)


def get_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    **model_params,
) -> Dict[str, str]:
    """_summary_

    Args:
        train_examples (List[Dict[str, Union[str, int]]]): _description_
        test_example (Dict[str, Union[str, int]]): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_

    Returns:
        Dict[str, str]: _description_
    """

    prompt_input, label = construct_prompt(
        train_examples, test_example, train_prompt_template, test_prompt_template
    )

    model_prediction = model_completion(prompt_input, model, **model_params)
    return {"prediction": model_prediction, "ground_truth": label}