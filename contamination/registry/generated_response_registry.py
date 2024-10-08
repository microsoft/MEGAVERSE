from contamination.utils.parse_generated_response_utils import (
    generate_xnli_str_from_generated_response,
    generate_xcopa_str_from_generated_response,
    generate_pawsx_str_from_generated_response,
    generate_tydiqa_str_from_generated_response,
    generated_udpos_str_from_generated_response,
)

GENERATED_RESPONSE_REGISTRY = {
    "xnli": generate_xnli_str_from_generated_response,
    "xcopa": generate_xcopa_str_from_generated_response,
    "paws-x": generate_pawsx_str_from_generated_response,
    "tydiqa": generate_tydiqa_str_from_generated_response,
    "udpos": generated_udpos_str_from_generated_response,
}
