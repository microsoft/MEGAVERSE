from contamination.templates import (
    INSTRUCTION_FOR_QUIZ_ANSWER,
    TEMPLATE_FOR_QUIZ_ANSWER,
)
from langchain.output_parsers import PydanticOutputParser
from mega.models.completion_models import model_completion
from mega.prompting.prompting_utils import get_substrate_prompt
import pandas as pd
import json
from typing import Dict, List, Union
from contamination.pydantic_models import AnswerResponse
from contamination.prompting_registry import GENERATED_RESPONSE_REGISTRY


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
        options=option_str,
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


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/t-sahuja/MultilingualBlanketEval/contamination/quizzes/xnli/dev-moonshot_rerun/test/ar/quiz_options.csv"
    )
    row = df.iloc[0]
    pydantic_parser = PydanticOutputParser(pydantic_object=AnswerResponse)
    prompt = create_quiz_answer_template(
        "xnli", "ar", row, "test", True, True, pydantic_parser
    ).strip()
    print(prompt)
    pass
