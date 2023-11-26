from pydantic import BaseModel, Field, validator
from typing import List


class XNLIResponse(BaseModel):
    Premise: str = Field(description="Premise in XNLI")
    Question: str = Field(description="Question in XNLI")
    Label: str = Field(description="Label in XNLI")


class AnswerResponse(BaseModel):
    answer: str = Field(description="Answer to the above multiple choice question")


class XNLIGeneratedResponse(BaseModel):
    options: List[XNLIResponse]

    @validator("options", pre=True)
    def validate_options(cls, v):
        if len(v) != 3:
            raise ValueError("There must be 3 options")

        for option in v:
            premise = option.get("Premise", None)
            question = option.get("Question", None)
            label = option.get("Label", None)
            if premise is None or question is None or label is None:
                raise ValueError("Premise, question, or label is missing")
        return v
