import os
import requests
import time
import openai
from openai import OpenAI

import google.generativeai as genai

from typing import List, Dict, Union, Any
from google.api_core.exceptions import ResourceExhausted
from promptsource.templates import Template
from mega.prompting.prompting_utils import construct_prompt
from mega.models.hf_completion_models import (
    hf_model_api_completion,
    hf_model_completion,
)
from mega.utils.const import (
    CHAT_MODELS,
    PALM_MAPPING,
    PALM_SUPPORTED_LANGUAGES_MAP,
    GEMINI_SUPPORTED_LANGUAGES_MAP,
    GEMINI_SAFETY_SETTINGS,
)
from mega.utils.env_utils import (
    load_openai_env_variables,
    HF_API_KEY,
    BLOOMZ_API_URL,
    HF_API_URL,
)
import backoff
from transformers import AutoModelForCausalLM, AutoTokenizer

from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt

load_openai_env_variables()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

client = OpenAI()

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=300)
def palm_api_completion(
    prompt: str, model: str = "text-bison@001", lang: str = "", **model_params
) -> str:
    if lang == "":
        raise ValueError("Language argument is necessary for palm model")
    if (
        lang not in PALM_SUPPORTED_LANGUAGES_MAP.keys()
        and lang not in PALM_SUPPORTED_LANGUAGES_MAP.values()
    ):
        raise ValueError("Language not supported by PALM!")

    if model == "text-bison-32k":
        from vertexai.preview.language_models import TextGenerationModel

        model_load = TextGenerationModel.from_pretrained(model)
    else:
        from vertexai.language_models import TextGenerationModel

        model_load = TextGenerationModel.from_pretrained(model)

    response = model_load.predict(
        prompt=prompt,
        max_output_tokens=model_params.get("max_tokens", 20),
        temperature=model_params.get("temperature", 1),
    )

    return response.text


def gemini_completion(
    prompt: str, model: str = "gemini-pro", lang: str = "", **model_params
) -> str:

    if lang == "":
        raise ValueError("Language argument is necessary for gemini model")
    if (
        lang not in GEMINI_SUPPORTED_LANGUAGES_MAP.keys()
        and lang not in GEMINI_SUPPORTED_LANGUAGES_MAP.values()
    ):
        raise ValueError("Language not supported by Gemini-Pro!")
    print(prompt)
    model_load = genai.GenerativeModel(model)
    response = model_load.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=model_params.get("temperature", 1),
            max_output_tokens=model_params.get("max_tokens", 50),
        ),
        safety_settings=GEMINI_SAFETY_SETTINGS,
    )

    try:
        return response.text
    except:
        print("Skipping")
        return ""


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_time=300,
)
def gpt3x_completion(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    run_details: Any = {},
    num_evals_per_sec: int = 2,
    **model_params,
) -> str:
    output = None
    if isinstance(prompt, str):
        response = client.completions.create(
            engine=model,
            prompt=prompt,
            max_tokens=model_params.get("max_tokens", 20),
            temperature=model_params.get("temperature", 1),
            top_p=model_params.get("top_p", 1),
        )
        if "num_calls" in run_details:
            run_details["num_calls"] += 1
        output = response.choices[0].text.strip().split("\n")[0]
        time.sleep(1 / num_evals_per_sec)
    else:
        response = client.chat.completions.create(
            engine=model,
            messages=prompt,
            max_tokens=model_params.get("max_tokens", 20),
            temperature=model_params.get("temperature", 1),
            top_p=model_params.get("top_p", 1),
        )
        if "num_calls" in run_details:
            run_details["num_calls"] += 1
        if response.choices[0].finish_reason == "content_filter":
            output = ""
        else:
            output = response.choices[0].message.content.strip()
        time.sleep(1 / num_evals_per_sec)

    return output


def bloom_completion(prompt: str, **model_params) -> str:
    """Runs the prompt over BLOOM model for text completion

    Args:
        prompt (str): Prompt String to be completed by the model

    Returns:
        str: generated string
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(payload):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        return response.json()

    output = ""
    while True:
        try:
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            break
        except Exception:
            if (
                "error_" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.InvalidRequestError
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            time.sleep(60)
            continue

    return output


def bloomz_completion(prompt: str, **model_params) -> str:
    """Runs the prompt over BLOOM model for text completion

    Args:
        prompt (str): Prompt String to be completed by the model

    Returns:
        str: generated string
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(payload):
        payload = {"inputs": payload}
        response = requests.post(BLOOMZ_API_URL, headers=headers, json=payload)
        return response.json()

    output = ""
    while True:
        try:
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            output = output.strip()
            break
        except Exception:
            if (
                "error" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.InvalidRequestError(
                    model_output["error"], model_output["error_type"]
                )
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            time.sleep(60)
            continue

    return output


def llama2_completion(prompt: str, model: str, **model_params) -> str:
    """Runs the prompt over LLAMA model for text completion

    Args:
        prompt (str): Prompt String to be completed by the model

    Returns:
        str: generated string
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(payload):
        payload = {"inputs": payload}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
        )
        return response.json()

    output = ""
    while True:
        try:
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            output = output.strip()
            break
        except Exception:
            if (
                "error" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.InvalidRequestError(
                    model_output["error"], model_output["error_type"]
                )
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            time.sleep(60)
            continue

    return output


def model_completion(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    lang: str,
    timeout: int = 0,
    model_obj: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
    **model_params,
) -> str:
    """Runs the prompt over one of the `SUPPORTED_MODELS` for text completion

    Args:
        - prompt (Union[str, List[Dict[str, str]]]) : Prompt String to be completed by the model
        - model (str) : Model to use

    Returns:
        str: generated string
    """

    if model_obj is not None and tokenizer is not None:
        return hf_model_completion(
            prompt,
            model_obj=model_obj,
            tokenizer=tokenizer,
            **model_params,
        )

    elif model in CHAT_MODELS:
        return gpt3x_completion(prompt, model, timeout=timeout, **model_params)

    elif model == "BLOOM":
        return bloom_completion(prompt, **model_params)

    elif model == "BLOOMZ":
        return bloomz_completion(prompt, **model_params)

    elif "Llama-2" in model:
        return hf_model_api_completion(prompt, model, **model_params)
    elif "palm" in model:
        return palm_api_completion(
            prompt, model=PALM_MAPPING[model], lang=lang, **model_params
        )

    if "gemini" in model:
        return gemini_completion(prompt, model=model, lang=lang, **model_params)


def get_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    lang: str,
    model_obj: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
    chat_prompt: bool = False,
    instruction: str = "",
    timeout: int = 0,
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
        train_examples,
        test_example,
        train_prompt_template,
        test_prompt_template,
        chat_prompt=(chat_prompt and model in CHAT_MODELS),
        instruction=instruction,
    )

    if model_obj is not None and tokenizer is not None and chat_prompt:
        prompt_input = convert_to_hf_chat_prompt(prompt_input, model)

    model_prediction = model_completion(
        prompt_input,
        model,
        lang,
        timeout=timeout,
        model_obj=model_obj,
        tokenizer=tokenizer,
        **model_params,
    )

    return {"prediction": model_prediction, "ground_truth": label}
