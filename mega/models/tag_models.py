import time
import warnings
from typing import Any, Dict, List, Union
import backoff
import openai
from openai import OpenAI

import os
import requests
import google.generativeai as genai

from google.api_core.exceptions import ResourceExhausted
from mega.utils.const import (
    PALM_SUPPORTED_LANGUAGES_MAP,
    CHAT_MODELS,
    PALM_MAPPING,
    GEMINI_SUPPORTED_LANGUAGES_MAP,
    GEMINI_SAFETY_SETTINGS,
)

from mega.prompting.prompting_utils import construct_tagging_prompt
from mega.utils.env_utils import (
    BLOOMZ_API_URL,
    HF_API_KEY,
    HF_API_URL,
    load_openai_env_variables,
)

load_openai_env_variables()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

client = OpenAI()


udpos_verbalizer = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating-conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper-noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating-conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

panx_verbalizer = {
    "B-PER": "begin-person",
    "I-PER": "inside-person",
    "B-ORG": "begin-organization",
    "I-ORG": "inside-organization",
    "B-LOC": "begin-location",
    "I-LOC": "inside-location",
    "O": "non-entity",
}

def gpt3x_tagger(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    test_tokens: List[str],
    delimiter: str = "_",
    num_evals_per_second: int = 2,
    one_shot_tag: bool = True,
    run_details: Any = {},
    num_evals_per_sec: int = 2,
    backoff_base: int = 2,
    backoff_rate: int = 2,
    backoff_ceil: int = 10,
    **model_params,
) -> str:
    chat_prompt = isinstance(prompt, list)

    if chat_prompt and not one_shot_tag:
        raise ValueError(
            "Chat Completion not supported for iterative tagging. Either set one_shot_tag = True or chat_prompts = False"
        )

    @backoff.on_exception(
        backoff.expo,
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
        ),
        max_time=60,
    )
    def predict_tag(prompt, token):
        prompt_with_token = f"{prompt} {token}{delimiter}"
        # Hit the api repeatedly till response is obtained
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt_with_token,
                max_tokens=model_params.get("max_tokens", 20),
                temperature=model_params.get("temperature", 1),
                top_p=model_params.get("top_p", 1),
            )

        except TypeError:
            warnings.warn(
                "Couldn't generate response, returning empty string as response"
            )
            return ""
        # import pdb
        # pdb.set_trace()
        return response.choices[0].text.strip().split()[0]

    @backoff.on_exception(
        backoff.expo,
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
        ),
        max_time=60,
    )
    def predict_one_shot():
        output = None
        try:
            if isinstance(prompt, str):
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=model_params.get("max_tokens", 20),
                    temperature=model_params.get("temperature", 1),
                    top_p=model_params.get("top_p", 1),
                )
                if "num_calls" in run_details:
                    run_details["num_calls"] += 1
                output = response.choices[0].text.strip().split("\n")[0]
            else:
                response = client.chat.completions.create(
                    model=model,
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
                    output = response.choices[0].message.content.strip().split("\n")[0]
        except TypeError:
            warnings.warn(
                "Couldn't generate response, returning empty string as response"
            )
            return ""

        return output

    if model in CHAT_MODELS:
        openai.api_version = "2023-03-15-preview"
    else:
        openai.api_version = "2022-12-01"

    if one_shot_tag:
        predicted_tokens_wth_tags = predict_one_shot()
        predicted_tokens_wth_tags = predicted_tokens_wth_tags.split()
        predicted_tags = []
        for i, token in enumerate(test_tokens):
            if i >= len(predicted_tokens_wth_tags):
                predicted_tags.append("")
                continue
            pred_token_nd_tag = predicted_tokens_wth_tags[i].split(delimiter)
            if len(pred_token_nd_tag) == 2:
                pred_token, pred_tag = pred_token_nd_tag
            else:
                pred_token = ""
                pred_tag = ""
            if token == pred_token:
                predicted_tags.append(pred_tag)
            else:
                predicted_tags.append("")
        return predicted_tags
    else:
        prompt_with_decodings = prompt
        predicted_tags = []
        for token in test_tokens:
            predicted_tag = predict_tag(prompt_with_decodings, token)
            prompt_with_decodings += f" {token}{delimiter}{predicted_tag}"
            predicted_tags.append(predicted_tag)
        return predicted_tags


@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=300)
def palm_tagger(
    prompt: str,
    model: str = "text-bison@001",
    lang: str = "",
    test_tokens: List[str] = [],
    one_shot_tag: bool = True,
    delimiter: str = "_",
    **model_params,
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

    def predict_tag(prompt, token):
        prompt_with_token = f"{prompt} {token}{delimiter}"
        model_output = model_load.predict(
            prompt_with_token,
            max_output_tokens=model_params.get("max_tokens", 20),
            temperature=model_params.get("temperature", 1),
        )
        return model_output.text.strip().split()[0]

    def predict_one_shot():
        model_output = model_load.predict(
            prompt,
            max_output_tokens=model_params.get("max_tokens", 20),
            temperature=model_params.get("temperature", 1),
        )

        return model_output.text

    if one_shot_tag:
        predicted_tokens_wth_tags = predict_one_shot()
        predicted_tokens_wth_tags = predicted_tokens_wth_tags.split()
        predicted_tags = []
        for i, token in enumerate(test_tokens):
            if i >= len(predicted_tokens_wth_tags):
                predicted_tags.append("")
                continue
            pred_token_nd_tag = predicted_tokens_wth_tags[i].split(delimiter)
            if len(pred_token_nd_tag) == 2:
                pred_token, pred_tag = pred_token_nd_tag
            else:
                pred_token = ""
                pred_tag = ""
            if token == pred_token:
                predicted_tags.append(pred_tag)
            else:
                predicted_tags.append("")
        return predicted_tags

    predicted_tags = []
    prompt_with_decodings = prompt

    for token in test_tokens:
        predicted_tag = predict_tag(prompt_with_decodings, token)
        prompt_with_decodings += f" {token}{delimiter}{predicted_tag}"
        predicted_tags.append(predicted_tag)
    return predicted_tags


@backoff.on_exception(backoff.expo, Exception, max_time=300)
def gemini_tagger(
    prompt: str,
    model: str = "gemini-pro",
    lang: str = "",
    test_tokens: List[str] = [],
    one_shot_tag: bool = True,
    delimiter: str = "_",
    **model_params,
) -> str:
    if lang == "":
        raise ValueError("Language argument is necessary for Gemini model")
    if (
        lang not in GEMINI_SUPPORTED_LANGUAGES_MAP.keys()
        and lang not in GEMINI_SUPPORTED_LANGUAGES_MAP.values()
    ):
        raise ValueError("Language not supported by GEMINI!")

    model_load = genai.GenerativeModel(model)

    def predict_tag(prompt, token):
        prompt_with_token = f"{prompt} {token}{delimiter}"
        model_output = model_load.generate_content(
            prompt_with_token,
            generation_config=genai.types.GenerationConfig(
                temperature=model_params.get("temperature", 1),
                max_output_tokens=model_params.get("max_tokens", 50),
            ),
            safety_settings=GEMINI_SAFETY_SETTINGS,
        )

        try:
            return model_output.text
        except Exception as e:
            print("Skipping due to error: ", e)
            return ""

    def predict_one_shot():
        model_output = model_load.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=model_params.get("temperature", 1),
                max_output_tokens=model_params.get("max_tokens", 50),
            ),
            safety_settings=GEMINI_SAFETY_SETTINGS,
        )

        try:
            return model_output.text
        except Exception as e:
            print("Skipping due to error: ", e)
            return ""

    if one_shot_tag:
        predicted_tokens_wth_tags = predict_one_shot()
        predicted_tokens_wth_tags = predicted_tokens_wth_tags.split()
        predicted_tags = []
        for i, token in enumerate(test_tokens):
            if i >= len(predicted_tokens_wth_tags):
                predicted_tags.append("")
                continue
            pred_token_nd_tag = predicted_tokens_wth_tags[i].split(delimiter)
            if len(pred_token_nd_tag) == 2:
                pred_token, pred_tag = pred_token_nd_tag
            else:
                pred_token = ""
                pred_tag = ""
            if token == pred_token:
                predicted_tags.append(pred_tag)
            else:
                predicted_tags.append("")
        return predicted_tags

    predicted_tags = []
    prompt_with_decodings = prompt

    for token in test_tokens:
        predicted_tag = predict_tag(prompt_with_decodings, token)
        prompt_with_decodings += f" {token}{delimiter}{predicted_tag}"
        predicted_tags.append(predicted_tag)
    return predicted_tags


def bloom_tagger(
    prompt: str,
    model: str,
    test_tokens: List[str],
    delimiter: str = "_",
    **model_params,
) -> str:
    assert model in ["BLOOM", "BLOOMZ"]

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(payload):
        if model == "bloom":
            payload = payload
            url = HF_API_URL
        else:
            payload = {"inputs": payload}
            url = BLOOMZ_API_URL
        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def predict_tag(prompt, token):
        prompt_with_token = f"{prompt} {token}{delimiter}"

        # Hit the api repeatedly till response is obtained
        output = ""
        while True:
            try:
                model_output = query(prompt_with_token)
                output = (
                    model_output[0]["generated_text"][len(prompt_with_token) :]
                    .strip()
                    .split()[0]
                )
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

    prompt_with_decodings = prompt
    predicted_tags = []
    for token in test_tokens:
        predicted_tag = predict_tag(prompt_with_decodings, token)
        prompt_with_decodings += f" {token}{delimiter}{predicted_tag}"
        predicted_tags.append(predicted_tag)

    return predicted_tags


def model_tagger(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    lang: str,
    test_tokens: List[str],
    delimiter: str = "_",
    num_evals_per_second: int = 2,
    one_shot_tag: bool = True,
    run_details: Any = {},
    **model_params,
) -> str:
    if model in CHAT_MODELS:
        return gpt3x_tagger(
            prompt,
            model,
            test_tokens,
            delimiter,
            num_evals_per_second=num_evals_per_second,
            one_shot_tag=one_shot_tag,
            run_details=run_details,
            **model_params,
        )
    elif model in ["BLOOM", "BLOOMZ"]:
        return bloom_tagger(
            prompt,
            model,
            test_tokens,
            delimiter,
            **model_params,
        )

    elif "palm" in model:
        return palm_tagger(
            prompt,
            PALM_MAPPING[model],
            lang,
            test_tokens,
            delimiter,
            **model_params,
        )

    elif "gemini-pro" in model:
        return gemini_tagger(
            prompt,
            model,
            lang,
            test_tokens,
            delimiter,
            **model_params,
        )

def get_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    prompt_template: str,
    verbalizer: Dict[str, str],
    model: str,
    lang: str,
    delimiter: str = "_",
    num_evals_per_second: int = 2,
    chat_prompt: bool = False,
    instruction: str = "",
    one_shot_tag: bool = True,
    run_details: Any = {},
    **model_params,
):
    reverse_verbalizer = {value: key for key, value in verbalizer.items()}

    prompt_input, label = construct_tagging_prompt(
        train_examples,
        test_example,
        prompt_template,
        verbalizer,
        delimiter=delimiter,
        chat_prompt=chat_prompt,
        instruction=instruction,
    )
    model_prediction = model_tagger(
        prompt_input,
        model,
        lang,
        test_tokens=test_example["tokens"],
        delimiter=delimiter,
        num_evals_per_second=num_evals_per_second,
        chat_prompt=chat_prompt,
        one_shot_tag=one_shot_tag,
        run_details=run_details,
        **model_params,
    )
    model_prediction_tags = [
        reverse_verbalizer.get(prediction_tag, prediction_tag)
        for prediction_tag in model_prediction
    ]

    return {"prediction": model_prediction_tags, "ground_truth": label}
