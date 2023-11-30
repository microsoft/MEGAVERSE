from contamination.templates import (
    INSTRUCTION_FOR_QUIZ_ANSWER,
    TEMPLATE_FOR_QUIZ_ANSWER,
)
from langchain.output_parsers import PydanticOutputParser
from mega.models.completion_models import model_completion
from mega.prompting.prompting_utils import get_substrate_prompt
from mega.utils.substrate_llm import LLMClient

import pandas as pd
import sys
import yaml
import os
import json
from tqdm import tqdm
from typing import Dict, List, Union
from contamination.pydantic_models import AnswerResponse
from contamination.registry.generated_response_registry import (
    GENERATED_RESPONSE_REGISTRY,
)


def create_quiz_answer_template(
    dataset_name: str,
    lang: str,
    row_df: pd.Series,
    dataset_split: str,
    chat_prompt: bool,
    substrate_prompt: bool,
    pydantic_parser: PydanticOutputParser,
) -> Union[str, List[Dict[str, str]]]:
    """Create a quiz answer template for the given dataset, language, and template.

    Args:
        dataset_name (str): Name of the dataset
        lang (str): Language of the dataset
        template_name (str): Name of the template
        dataset_split (str): Split of the dataset
        chat_prompt (bool): Whether to use a chat prompt
        substrate_prompt (bool): Whether to use a substrate prompt
        pydantic_parser (PydanticOutputParser): Pydantic parser

    Returns:
        str: Quiz answer template
    """
    if dataset_name not in GENERATED_RESPONSE_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not supported")
    generated_response = json.loads(row_df["generated_response"])["options"]
    original_response = row_df["original_example"]
    option_str = GENERATED_RESPONSE_REGISTRY[dataset_name](generated_response)

    option_str += "D) " + original_response.strip() + "\n"

    format_instructions = pydantic_parser.get_format_instructions()

    instruction = INSTRUCTION_FOR_QUIZ_ANSWER.format(
        dataset=dataset_name, lang=lang, dataset_split=dataset_split
    )
    prompt = TEMPLATE_FOR_QUIZ_ANSWER.format(
        instruction=instruction if not chat_prompt else "",
        options=option_str.strip(),
        format_instructions=format_instructions,
    )

    if not chat_prompt:
        return prompt.strip()

    else:
        messages = [
            {"role": "system", "content": instruction.strip()},
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ]

        if substrate_prompt:
            if substrate_prompt and not chat_prompt:
                raise ValueError(
                    "Cannot use substrate prompt without chat prompt. Please set chat_prompt=True"
                )

            prompt = get_substrate_prompt(messages)
            return prompt.strip()
        else:
            return messages


def kappa_fixed_value(observed_agreement_probability: float) -> float:
    kappa_fixed = (observed_agreement_probability - 0.25) / 0.75
    return kappa_fixed


def calculate_contamination(results_df: pd.DataFrame, **kwargs):
    total_correct = 0
    total = len(results_df)

    select_samples = min(100, total)

    for idx, row in results_df.sample(select_samples).iterrows():
        answer = json.loads(row["answer"])["answer"].strip()
        if answer.upper() == "D":
            total_correct += 1

    score = total_correct / select_samples
    contamination = kappa_fixed_value(score)
    out_dir = kwargs["out_dir"]
    kwargs["score"] = score
    kwargs["contamination"] = contamination
    
    # dump kwargs, score and contamination into a json file in out_dir
    with open(f"{out_dir}/contamination.json", "w") as f:
        json.dump(kwargs, f)


def get_quiz_answers(
    dataset_name: str,
    dataset_split: str,
    model_name: str,
    lang: str,
    out_dir: str,
    df: pd.DataFrame,
    chat_prompt: bool,
    substrate_prompt: bool,
    pydantic_parser: PydanticOutputParser,
    max_tokens: int,
    temperature: float,
    llm_client: LLMClient = None,
):
    out_dir = f"{out_dir}/{lang}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "quiz_answers.csv" in os.listdir(out_dir):
        results_df = pd.read_csv(f"{out_dir}/quiz_answers.csv")
        results = results_df.to_dict("records")

    else:
        results = []

    pred_len = len(results)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if idx < pred_len:
            print(f"skipping {idx}")
            continue
        prompt = create_quiz_answer_template(
            dataset_name,
            lang,
            row,
            dataset_split,
            chat_prompt,
            substrate_prompt,
            pydantic_parser,
        ).strip()

        answer = model_completion(
            prompt,
            model_name,
            lang,
            run_substrate_llm_completion=substrate_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            llm_client=llm_client,
        )
        try:
            parsed_response = pydantic_parser.parse(answer)
            results.append(
                {
                    "answer": parsed_response.json(),
                    "prompt": prompt,
                }
            )

        except ValueError as e:
            print(f"Error for {idx}")
            print(e)
            continue
        results_df = pd.DataFrame(results)

        results_df.to_csv(f"{out_dir}/quiz_answers.csv", index=False)
    
    calculate_contamination(
        results_df=results_df,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        model_name=model_name,
        lang=lang,
        out_dir=out_dir,
        chat_prompt=chat_prompt,
        substrate_prompt=substrate_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


if __name__ == "__main__":
    args_path = sys.argv[1]
    if not os.path.exists(args_path):
        raise ValueError(f"{args_path} does not exist")

    with open(args_path, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)

    dataset_name = args["dataset_name"]
    save_dir = args["save_dir"]
    quiz_dir = args["quiz_dir"]
    model_name = args["model_name"]
    max_tokens = args["max_tokens"]
    temperature = args["temperature"]
    template_name = args["template_name"]
    langs = args["langs"]
    dataset_split = args["dataset_split"]
    chat_prompt = args["chat_prompt"]
    substrate_prompt = args["substrate_prompt"]
    out_dir = f"{save_dir}/{dataset_name}/{model_name}/{dataset_split}"
    llm_client = LLMClient() if substrate_prompt else None
    quiz_dir = f"{quiz_dir}/{dataset_name}/{model_name}/{dataset_split}"
    quiz_dir = args["quiz_dir"]
    pydantic_parser = PydanticOutputParser(pydantic_object=AnswerResponse)
    for lang in langs:
        quiz_path = f"{quiz_dir}/{lang}/quiz_options.csv"
        print("Generating quiz answers for", lang)
        if not os.path.exists(quiz_path):
            print(
                f"Either {quiz_path} does not exist or {lang} is not supported by PaLM2"
            )
            continue

        df = pd.read_csv(quiz_path)

        get_quiz_answers(
            dataset_name,
            dataset_split,
            model_name,
            lang,
            out_dir,
            df,
            chat_prompt,
            substrate_prompt,
            pydantic_parser,
            max_tokens,
            temperature,
            llm_client,
        )
