from contamination.pydantic_models import (
    XNLIGeneratedResponse,
    PAWSXGeneratedResponse,
    XCOPAGeneratedResponse,
    UDPOSGeneratedResponse,
    tydiqaGeneratedResponse
)

PYDANTIC_REGISTRY = {
    "xnli": XNLIGeneratedResponse,
    "paws-x": PAWSXGeneratedResponse,
    "xcopa": XCOPAGeneratedResponse,
    "udpos": UDPOSGeneratedResponse,
    "tydiqa": tydiqaGeneratedResponse
}
