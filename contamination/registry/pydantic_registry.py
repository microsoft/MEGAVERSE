<<<<<<< HEAD
from contamination.pydantic_models import XNLIGeneratedResponse, PAWSXGeneratedResponse

PYDANTIC_REGISTRY = {
    "xnli": XNLIGeneratedResponse,
    'paws-x': PAWSXGeneratedResponse,
=======
from contamination.pydantic_models import XNLIGeneratedResponse, XCOPAGeneratedResponse

PYDANTIC_REGISTRY = {
    "xnli": XNLIGeneratedResponse,
    "xcopa": XCOPAGeneratedResponse,
>>>>>>> 134eb21c20e93d0cdd897bef366f5d75da4b1266
}
