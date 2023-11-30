from contamination.pydantic_models import XNLIGeneratedResponse, PAWSXGeneratedResponse

PYDANTIC_REGISTRY = {
    "xnli": XNLIGeneratedResponse,
    'paws-x': PAWSXGeneratedResponse,
}
