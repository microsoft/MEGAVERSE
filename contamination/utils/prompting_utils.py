from contamination.templates import VERBALIZER_XNLI, VERBALIZER_PAWSX
from mega.prompting.prompting_utils import get_substrate_prompt
from typing import Dict, List
from contamination.registry.langs_registry import LANGS
from contamination.utils.general_utils import render_jinja_template


def generate_gpt4_style_prompt(
    instruction: str, prompt: str, substrate_prompt: bool, lang: str
):
    messages = [
        {"role": "system", "content": instruction.format(lang=LANGS[lang])},
        {"role": "user", "content": prompt},
    ]

    if substrate_prompt:
        sub_prompt = get_substrate_prompt(messages)
        return sub_prompt
    else:
        return messages


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
        return generate_gpt4_style_prompt(instruction, prompt, substrate_prompt, lang)


def get_xcopa_quiz_generation_prompt(
    dataset_example: Dict[str, any],
    template: str,
    instruction: str,
    chat_prompt: bool,
    substrate_prompt: bool,
    lang: str,
    format_instructions: str,
) -> str:
    
    
    rendered_prompt = render_jinja_template(prompt, dataset_example)
    
    prompt = rendered_prompt.format(
        instruction=instruction.format(lang=LANGS[lang]) if not chat_prompt else "",
        format_instructions=format_instructions,
    )
    dataset_example["answer_choices"] = {
        0: "choice1",
        1: "choice2",
    }

    if not chat_prompt:
        return prompt
    else:
        return generate_gpt4_style_prompt(
            instruction, prompt, substrate_prompt, lang
        )

def get_pawsx_quiz_generation_prompt(
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
        sentence1=dataset_example["sentence1"],
        sentence2=dataset_example["sentence2"],
        label=dataset_example["label"],
        verbalized_label=VERBALIZER_PAWSX[dataset_example["label"]],
        format_instructions=format_instructions,
    )

    if not chat_prompt:
        return prompt

    else:
        return generate_gpt4_style_prompt(instruction, prompt, substrate_prompt, lang)