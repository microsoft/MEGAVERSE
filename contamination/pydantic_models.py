from pydantic import BaseModel, Field, validator
from typing import List


class AnswerResponse(BaseModel):
    answer: str = Field(description="Answer to the above multiple choice question")


class XNLIResponse(BaseModel):
    Premise: str = Field(description="Premise in XNLI")
    Question: str = Field(description="Question in XNLI")
    Label: str = Field(description="Label in XNLI")


class XCOPAResponse(BaseModel):
    premise: str = Field(description="Premise in XCOPA")
    choice1: str = Field(description="Choice 1 in XCOPA")
    choice2: str = Field(description="Choice 2 in XCOPA")
    answer: str = Field(description="Answer in XCOPA")


class XCOPAGeneratedResponse(BaseModel):
    options: List[XCOPAResponse]

    @validator("options", pre=True)
    def validate_options(cls, v):
        if len(v) != 3:
            raise ValueError("There must be 3 options")

        for option in v:
            premise = option.get("premise", None)
            choice1 = option.get("choice1", None)
            choice2 = option.get("choice2", None)
            answer = option.get("answer", None)
            if premise is None or choice1 is None or choice2 is None or answer is None:
                raise ValueError("Premise, choice1, choice2, or answer is missing")
        return v


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
