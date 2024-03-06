from pydantic import BaseModel, Field, validator
from typing import List


class AnswerResponse(BaseModel):
    answer: str = Field(
        description="Answer to the above multiple choice question. Answer must either be A, B, C, or D"
    )

    @validator("answer", pre=True)
    def validate_answer(cls, v):
        if v not in ["A", "B", "C", "D"]:
            raise ValueError("Answer must be A, B, C, or D")
        return v


class XNLIResponse(BaseModel):
    Premise: str = Field(description="Premise in XNLI")
    Question: str = Field(description="Question in XNLI")
    Label: str = Field(description="Label in XNLI")


class XCOPAResponse(BaseModel):
    premise: str = Field(description="Premise in XCOPA")
    choice1: str = Field(description="Choice 1 in XCOPA")
    choice2: str = Field(description="Choice 2 in XCOPA")
    label: str = Field(description="Answer in XCOPA")


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
            label = option.get("label", None)
            if premise is None or choice1 is None or choice2 is None or label is None:
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


class PAWSXResponse(BaseModel):
    Sentence1: str = Field(description="Sentence 1 in pawsx")
    Sentence2: str = Field(description="Sentence 2 in pawsx")
    Label: str = Field(description="Label in pawsx")


class PAWSXGeneratedResponse(BaseModel):
    options: List[PAWSXResponse]

    @validator("options", pre=True)
    def validate_options(cls, v):
        if len(v) != 3:
            raise ValueError("There must be 3 options")

        for option in v:
            sentence1 = option.get("Sentence1", None)
            sentence2 = option.get("Sentence2", None)
            label = option.get("Label", None)
            if sentence1 is None or sentence2 is None or label is None:
                raise ValueError("sentence1, sentence2, or label is missing")
        return v


class tydiqaResponse(BaseModel):
    context: str = Field(description="context in tydiqa")
    question: str = Field(description="question in tydiqa")
    answer: dict = Field(description="answer in tydiqa")


class tydiqaGeneratedResponse(BaseModel):
    options: List[tydiqaResponse]

    @validator("options", pre=True)
    def validate_options(cls, v):
        if len(v) != 3:
            raise ValueError("There must be 3 options")
        for option in v:
            context = option.get("context", None)
            question = option.get("question", None)
            answer = str(option.get("answer", None))
            if context is None or question is None or answer is None:
                raise ValueError("context, question, or answer is missing")
        # print(contex)
        return v


class UDPOSResponse(BaseModel):
    tokens: str = Field(description="Tokens in UDPOS")
    tags: str = Field(description="Tags in UDPOS")
    tagged_tokens: str = Field(description="Tagged tokens in UDPOS")


class UDPOSGeneratedResponse(BaseModel):
    options: List[UDPOSResponse]

    @validator("options", pre=True)
    def validate_options(cls, v):
        if len(v) != 3:
            raise ValueError("There must be 3 options")

        for option in v:
            tokens = option.get("tokens", None)
            tags = option.get("tags", None)
            tagged_tokens = option.get("tagged_tokens", None)
            if tokens is None or tags is None or tagged_tokens is None:
                raise ValueError("tokens, tags, or tagged_tokens is missing")
        return v
