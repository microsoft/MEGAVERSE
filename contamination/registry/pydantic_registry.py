from contamination.pydantic_models import XNLIGeneratedResponse, PAWSXGeneratedResponse, XCOPAGeneratedResponse, UDPOSGeneratedResponse

PYDANTIC_REGISTRY = {
    "xnli": XNLIGeneratedResponse,
    'paws-x': PAWSXGeneratedResponse,
    "xcopa": XCOPAGeneratedResponse,
    "udpos": UDPOSGeneratedResponse
}
