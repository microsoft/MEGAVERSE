from contamination.utils.prompting_utils import (
    get_xnli_quiz_generation_prompt,
    get_xcopa_quiz_generation_prompt,
)

QUIZ_GENERATION_PROMPT_REGISTRY = {
    "xnli": get_xnli_quiz_generation_prompt,
    "xcopa": get_xcopa_quiz_generation_prompt
}
