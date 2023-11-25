from contamination.pydantic_models import XNLIGeneratedResponse
from contamination.prompting_utils import get_xnli_quiz_generation_prompt

PYDANTIC_DICT = {
    "xnli": XNLIGeneratedResponse,
}

# QUIZ_GENERATION_PROMPT_UTILS = {
#     "xnli": get_xnli_quiz_generation_prompt,
# }
