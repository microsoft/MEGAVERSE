from contamination.prompting_utils import (
    get_xnli_quiz_generation_prompt,
    generate_xnli_str_from_generated_response,
)

QUIZ_GENERATION_PROMPT_REGISTRY = {
    "xnli": get_xnli_quiz_generation_prompt,
}

GENERATED_RESPONSE_REGISTRY = {
    "xnli": generate_xnli_str_from_generated_response,
}
