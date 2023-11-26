from contamination.parse_generated_response_utils import (
    generate_xnli_str_from_generated_response,
)

GENERATED_RESPONSE_REGISTRY = {
    "xnli": generate_xnli_str_from_generated_response,
}
