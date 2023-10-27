import requests
import warnings
import signal
import time
import openai
from typing import List, Dict, Union, Any
from promptsource.templates import Template
from mega.prompting.prompting_utils import construct_prompt
from mega.utils.env_utils import (
    load_openai_env_variables,
    HF_API_KEY,
    # BLOOMZ_API_URL,
    HF_API_URL,
)
from huggingface_hub import InferenceClient
from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt

load_openai_env_variables()

SUPPORTED_MODELS = [
    "BLOOM",
    "BLOOMZ",
    "gpt-35-turbo",
    "gpt-4-32k",
    "gpt-4",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf"
]

MODEL_TYPES = [
                "completion", 
                "seq2seq"
               ]

CHAT_MODELS = [
    "gpt-35-turbo",
    "gpt4_deployment",
    "gpt-4",
    "gpt-4-32k",
    # "meta-llama/Llama-2-70b-chat-hf",
]


# Register an handler for the timeout
# def handler(signum, frame):
#     raise Exception("API Response Stuck!")

# signal.signal(signal.SIGALRM, handler)


def timeout_handler(signum, frame):
    raise openai.error.Timeout("API Response Stuck!")


def gpt3x_completion(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    run_details: Any = {},
    num_evals_per_sec: int = 2,
    backoff_base: int = 2,
    backoff_rate: int = 2,
    backoff_ceil: int = 10,
    timeout: int = 0,
    **model_params,
) -> str:
    """Runs the prompt over the GPT3.x model for text completion

    Args:
        - prompt (str) : Prompt String to be completed by the model
        - model (str) : GPT-3x model to use

    Returns:
        str: generated string
    """

    if model in CHAT_MODELS:
        openai.api_version = "2023-03-15-preview"
    else:
        openai.api_version = "2022-12-01"

    # Hit the api repeatedly till response is obtained
    output = None
    backoff_count = 0
    while True:
        try:
            if isinstance(prompt, str):
                if timeout != 0:
                    # Set the signal handler for the SIGALRM signa
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(
                        timeout
                    )  # Wait for timeout seconds for the response to come
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    max_tokens=model_params.get("max_tokens", 20),
                    temperature=model_params.get("temperature", 1),
                    top_p=model_params.get("top_p", 1),
                )
                if timeout != 0:
                    signal.alarm(0)  # Disable the alarm
                if "num_calls" in run_details:
                    run_details["num_calls"] += 1
                output = response["choices"][0]["text"].strip().split("\n")[0]
                time.sleep(1 / num_evals_per_sec)
                backoff_count = 0
            else:
                if timeout != 0:
                    # Set the signal handler for the SIGALRM signa
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(
                        timeout
                    )  # Wait for timeout seconds for the response to come
                response = openai.ChatCompletion.create(
                    engine=model,
                    messages=prompt,
                    max_tokens=model_params.get("max_tokens", 20),
                    temperature=model_params.get("temperature", 1),
                    top_p=model_params.get("top_p", 1),
                )
                if timeout != 0:
                    signal.alarm(0)  # Disable the alarm
                if "num_calls" in run_details:
                    run_details["num_calls"] += 1
                if response["choices"][0]["finish_reason"] == "content_filter":
                    output = ""
                else:
                    output = response["choices"][0]["message"][
                        "content"
                    ].strip()  # .split("\n")[0]
                time.sleep(1 / num_evals_per_sec)
                backoff_count = 0
            break
        except (openai.error.APIConnectionError, openai.error.RateLimitError) as e:
            backoff_count = min(backoff_count + 1, backoff_ceil)
            sleep_time = backoff_base**backoff_count
            print(f"Exceeded Rate Limit. Waiting for {sleep_time} seconds")
            time.sleep(sleep_time)
            continue
        except (openai.error.APIError, TypeError):
            warnings.warn(
                "Couldn't generate response, returning empty string as response"
            )
            return ""

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
            signal.alarm(60)  # Wait for a minute for the response to come
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            signal.alarm(0)  # Reset the alarm
            break
        except Exception as e:
            if (
                "error_" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.error.InvalidRequestError
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            signal.alarm(0)  # Reset the alarm
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
            # signal.alarm(60)  # Wait for a minute for the response to come
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            output = output.strip()
            # signal.alarm(0)  # Reset the alarm
            break
        except Exception as e:
            if (
                "error" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.error.InvalidRequestError(
                    model_output["error"], model_output["error_type"]
                )
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            # signal.alarm(0)  # Reset the alarm
            time.sleep(60)
            continue

    return output


def llama2_completion(prompt: str, model:str, **model_params) -> str:
    """Runs the prompt over BLOOM model for text completion

    Args:
        prompt (str): Prompt String to be completed by the model

    Returns:
        str: generated string
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(payload):
        payload = {"inputs": payload}
        response = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload)
        return response.json()

    output = ""
    while True:
        try:
            # signal.alarm(60)  # Wait for a minute for the response to come
            model_output = query(prompt)
            output = model_output[0]["generated_text"][len(prompt) :].split("\n")[0]
            output = output.strip()
            # signal.alarm(0)  # Reset the alarm
            break
        except Exception as e:
            if (
                "error" in model_output
                and "must have less than 1000 tokens." in model_output["error"]
            ):
                raise openai.error.InvalidRequestError(
                    model_output["error"], model_output["error_type"]
                )
            print("Exceeded Limit! Sleeping for a minute, will try again!")
            # signal.alarm(0)  # Reset the alarm
            time.sleep(60)
            continue

    return output


def model_completion(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    timeout: int = 0,
    **model_params,
) -> str:
    """Runs the prompt over one of the `SUPPORTED_MODELS` for text completion

    Args:
        - prompt (Union[str, List[Dict[str, str]]]) : Prompt String to be completed by the model
        - model (str) : Model to use

    Returns:
        str: generated string
    """

    print(model)
    if (
        model
        in CHAT_MODELS
    ):
        return gpt3x_completion(prompt, model, timeout=timeout, **model_params)

    if model == "BLOOM":
        return bloom_completion(prompt, **model_params)

    if model == "BLOOMZ":
        return bloomz_completion(prompt, **model_params)
    
    if "Llama-2" in model:
        print(prompt)
        
        prompt = llama2_completion(prompt, model, **model_params)
        


def get_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
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

    model_prediction = model_completion(
        prompt_input, model, timeout=timeout, **model_params
    )
    
    return {"prediction": model_prediction, "ground_truth": label}
