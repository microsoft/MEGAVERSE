import os
from mega.models.completion_models import model_completion
from mega.data.load_datasets import load_tagging_dataset
import yaml
import sys
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from mega.utils.substrate_llm import LLMClient
from contamination.registry.pydantic_registry import PYDANTIC_REGISTRY
from langchain.output_parsers import PydanticOutputParser
from contamination.templates import (
    INSTRUCTION_FOR_QUIZ_GENERATION,
    TEMPLATES,
)
from typing import Dict, Any, Union, List
from contamination.registry.prompting_registry import QUIZ_GENERATION_PROMPT_REGISTRY

import warnings

warnings.filterwarnings("ignore")


def construct_quiz_generation_prompt(
    dataset_name: str,
    pydantic_parser: PydanticOutputParser,
    dataset_example: Dict[str, Any],
    template: str,
    instruction: str,
    chat_prompt: bool = False,
    substrate_prompt: bool = False,
    lang: str = "en",
    **kwargs,
) -> Union[str, List[Dict[str, str]]]:
    """Constructs prompt for quiz generation

    Args:
        dataset_example (Dict[str, Any]): _description_
        template (str): _description_
        instruction (str): _description_

    Returns:
        str: _description_
    """
    if dataset_name in QUIZ_GENERATION_PROMPT_REGISTRY:
        format_instructions = pydantic_parser.get_format_instructions()

        prompt = QUIZ_GENERATION_PROMPT_REGISTRY[dataset_name](
            dataset_example,
            template,
            instruction,
            chat_prompt,
            substrate_prompt,
            lang,
            format_instructions,
            **kwargs,
        )

        original_example = (
            QUIZ_GENERATION_PROMPT_REGISTRY[dataset_name](
                dataset_example, template, "", False, False, lang, "", **kwargs
            )
            .replace("--TEXT--", "")
            .strip()
        )
        return prompt, original_example
    else:
        raise ValueError(f"{dataset_name} not supported")


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
    is_tagging_dataset: bool = False,
    llm_client: LLMClient = None,
    pydantic_parser: PydanticOutputParser = None,
):
    if is_tagging_dataset:

        def join_func(example):
            example["tokens"] = " ".join(example["tokens"]).strip()
            example["tagged_tokens"] = " ".join(example["tagged_tokens"]).strip()
            example["tags"] = " ".join(example["tags"]).strip()
            return example

        ds = load_tagging_dataset(dataset_name, lang, dataset_split)
        ds = ds.map(join_func)
    elif dataset_name == "tydiqa":

        TYDIQA_LANG2CODES = {
            "bengali": "bn",
            "korean": "ko",
            "swahili": "sw",
            "english": "en",
            "indonesian": "id",
            "arabic": "ar",
            "finnish": "fi",
            "telugu": "te",
            "russian": "ru",
        }
        ds = load_dataset(dataset_name, "secondary_task")[dataset_split]
        ds = ds.map(
            lambda example: {"lang": TYDIQA_LANG2CODES[example["id"].split("-")[0]]}
        )

    else:
        ds = load_dataset(dataset_name, lang)
        ds = ds[dataset_split]

    out_dir = f"{out_dir}/{lang}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "quiz_options.csv" in os.listdir(out_dir):
        results_df = pd.read_csv(f"{out_dir}/quiz_options.csv")
        results = results_df.to_dict("records")

    else:
        results = []

    num_points = min(num_points, len(ds))
    ds = ds.select(range(num_points))
    pbar = tqdm(ds)
    pred_len = len(results)
    for idx, test_example in enumerate(pbar):
        if idx < pred_len:
            print(f"skipping {idx}")
            continue
        prompt, original_example = construct_quiz_generation_prompt(
            dataset_name,
            pydantic_parser,
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
        try:
            parsed_response = pydantic_parser.parse(response)
            results.append(
                {
                    "generated_response": parsed_response.json(),
                    "original_example": original_example,
                }
            )

        except ValueError as e:
            print(f"Error for {idx}")
            print(e)
            continue

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{out_dir}/quiz_options.csv", index=False)


if __name__ == "__main__":
    # Parse args.yaml
    args_path = sys.argv[1]
    if not os.path.exists(args_path):
        raise ValueError(f"{args_path} does not exist")

    with open(args_path, "r") as file:
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
    is_tagging_dataset = args.get("is_tagging_dataset", False)

    out_dir = f"{save_dir}/{dataset_name}/{model_name}/{dataset_split}"
    llm_client = LLMClient() if substrate_prompt else None
    pydantic_parser = PydanticOutputParser(
        pydantic_object=PYDANTIC_REGISTRY[dataset_name]
    )

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
                is_tagging_dataset,
                llm_client,
                pydantic_parser,
            )

        except ValueError as e:
            print(f"Error for {lang}, not supported by Palm2")
            print(e)
            continue
        except KeyError:
            print(
                f"Language {lang} not registered in registry. Please register it. Skipping this for now"
            )
            continue
