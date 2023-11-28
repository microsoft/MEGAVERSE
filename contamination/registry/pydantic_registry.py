from contamination.pydantic_models import XNLIGeneratedResponse, XCOPAGeneratedResponse

PYDANTIC_REGISTRY = {
    "xnli": XNLIGeneratedResponse,
    "xcopa": XCOPAGeneratedResponse,
}
