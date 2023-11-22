from datasets import load_dataset
import os
from mega.models.completion_models import model_completion
import yaml
from mega.prompting.prompting_utils import get_substrate_prompt
from tqdm import tqdm
import pandas as pd
from mega.utils.substrate_llm import LLMClient
from contamination.templates import (
    INSTRUCTION_FOR_QUIZ_GENERATION,
    TEMPLATES,
    VERBALIZER_XNLI,
)
from typing import Dict, Any, List, Union, Tuple

# suppress warnings
import warnings

warnings.filterwarnings("ignore")


LANGS = {
    "hi": "Hindi",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "zh": "Chinese",
    "ar": "Arabic",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "ur": "Urdu",
    "sw": "Swahili",
    "bg": "Bulgarian",
    "el": "Greek",
    "sw": "Swahili",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ro": "Romanian",
    "pl": "Polish",
    "cs": "Czech",
    "da": "Danish",
}


def get_xnli_prompt(
    dataset_example: Dict[str, any],
    template: str,
    instruction: str,
    chat_prompt: bool,
    substrate_prompt: bool,
    lang: str,
) -> str:
    prompt = template.format(
        instruction=instruction.format(lang=LANGS[lang]) if not chat_prompt else "",
        premise=dataset_example["premise"],
        hypothesis=dataset_example["hypothesis"],
        label=dataset_example["label"],
        verbalized_label=VERBALIZER_XNLI[dataset_example["label"]],
    )

    if not chat_prompt:
        return prompt

    else:
        messages = [
            {"role": "system", "content": instruction.format(lang=LANGS[lang])},
            {
                "role": "user",
                "content": prompt,
            },
        ]

        if substrate_prompt:
            prompt = get_substrate_prompt(messages)
            return prompt
        else:
            return messages


def construct_quiz_generation_prompt(
    dataset_name: str,
    dataset_example: Dict[str, Any],
    template: str,
    instruction: str,
    chat_prompt: bool = False,
    substrate_prompt: bool = False,
    lang: str = "en",
) -> Union[str, List[Dict[str, str]]]:
    """Constructs prompt for quiz generation

    Args:
        dataset_example (Dict[str, Any]): _description_
        template (str): _description_
        instruction (str): _description_

    Returns:
        str: _description_
    """
    if dataset_name == "xnli":
        prompt = get_xnli_prompt(
            dataset_example,
            template,
            instruction,
            chat_prompt,
            substrate_prompt,
            lang,
        )
        original_example = get_xnli_prompt(
            dataset_example, template, "", False, False, lang
        ).replace("--TEXT--", "")
        return prompt, original_example

    return ""


def run_quiz_creation(
    dataset_name: str,
    lang: str,
    out_dir: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    template_name: str,
    dataset_split: str,
    chat_prompt: bool,
    substrate_prompt: bool,
    num_points: int = 100,
    llm_client: LLMClient = None,
):
    ds = load_dataset(dataset_name, lang)
    ds = ds[dataset_split]
    ds = ds.select(range(num_points))
    pbar = tqdm(ds)
    out_dir = f"{out_dir}/{lang}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "quiz_options.csv" in os.listdir(out_dir):
        results_df = pd.read_csv(f"{out_dir}/quiz_options.csv")
        results = results_df.to_dict("records")

    else:
        results = []

    pred_len = len(results)
    for idx, test_example in enumerate(pbar):
        if idx < pred_len:
            print(f"skipping {idx}")
            continue
        prompt, original_example = construct_quiz_generation_prompt(
            dataset_name,
            test_example,
            TEMPLATES[template_name],
            INSTRUCTION_FOR_QUIZ_GENERATION,
            chat_prompt=chat_prompt,
            substrate_prompt=substrate_prompt,
            lang=lang,
        )

        response = model_completion(
            prompt,
            model_name,
            lang,
            max_tokens=max_tokens,
            temperature=temperature,
            run_substrate_llm_completion=substrate_prompt,
            llm_client=llm_client,
        )

        results.append(
            {"generated_response": response, "original_example": original_example}
        )

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{out_dir}/quiz_options.csv", index=False)


if __name__ == "__main__":
    # Parse args.yaml
    with open("contamination/configs/xnli_palm_args.yaml", "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)

    dataset_name = args["dataset_name"]
    save_dir = args["save_dir"]
    model_name = args["model_name"]
    max_tokens = args["max_tokens"]
    temperature = args["temperature"]
    template_name = args["template_name"]
    langs = args["langs"]
    dataset_split = args["dataset_split"]
    chat_prompt = args["chat_prompt"]
    substrate_prompt = args["substrate_prompt"]
    num_points = args["num_points"]
    out_dir = f"{save_dir}/{dataset_name}/{model_name}/{dataset_split}"
    llm_client = LLMClient() if substrate_prompt else None

    # test_example = load_dataset(dataset_name, "en")[dataset_split][0]
    # lang = "en"
    # create_prompt = construct_quiz_generation_prompt(
    #     dataset_name,
    #     test_example,
    #     TEMPLATES[template_name],
    #     INSTRUCTION_FOR_QUIZ_GENERATION,
    #     chat_prompt=chat_prompt,
    #     substrate_prompt=substrate_prompt,
    #     lang=lang,
    # )

    # response = model_completion(
    #     create_prompt,
    #     model_name,
    #     lang,
    #     max_tokens=max_tokens,
    #     temperature=temperature,
    # )

    # print(response)

    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    for lang in langs:
        print("creating dataset for lang", lang)
        try:
            run_quiz_creation(
                dataset_name,
                lang,
                out_dir,
                model_name,
                max_tokens,
                temperature,
                template_name,
                dataset_split,
                chat_prompt,
                substrate_prompt,
                num_points,
                llm_client,
            )

        except ValueError as e:
            print(f"Error for {lang}, not supported by Palm2")
            print(e)
            continue
