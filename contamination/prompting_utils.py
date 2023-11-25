from contamination.templates import VERBALIZER_XNLI
from mega.prompting.prompting_utils import get_substrate_prompt
from typing import Dict, List
from contamination.langs_registry import LANGS


def get_xnli_quiz_generation_prompt(
    dataset_example: Dict[str, any],
    template: str,
    instruction: str,
    chat_prompt: bool,
    substrate_prompt: bool,
    lang: str,
    format_instructions: str,
) -> str:
    prompt = template.format(
        instruction=instruction.format(lang=LANGS[lang]) if not chat_prompt else "",
        premise=dataset_example["premise"],
        hypothesis=dataset_example["hypothesis"],
        label=dataset_example["label"],
        verbalized_label=VERBALIZER_XNLI[dataset_example["label"]],
        format_instructions=format_instructions,
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


def generate_xnli_str_from_generated_response(
    generated_response: List[Dict[str, str]]
) -> str:
    """Generate a string from the generated response.

    Args:
        generated_response (List[Dict[str,str]]): Generated response

    Returns:
        str: Generated string
    """
    generated_str = ""
    for i, option in enumerate(generated_response):
        generated_str += chr(ord("A") + i)
        generated_str += ") "
        generated_str += "Premise: " + option["Premise"].strip() + "\n"
        generated_str += "Question: " + option["Question"].strip() + "\n"
        generated_str += "Label: " + option["Label"].strip() + "\n"
        generated_str += "\n"
    return generated_str
