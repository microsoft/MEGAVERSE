from typing import Union, List, Dict, Tuple, Any
from promptsource.templates import Template, DatasetTemplates
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


def get_substrate_prompt(messages: List[Dict[str, str]]) -> str:
    prompt = ""
    pStart = "<|im_start|>"
    pEnd = "<|im_end|>"
    newLine = "\n"

    sz = len(messages) - 1
    for item in messages[:sz]:
        buf = f'{pStart}{item["role"]}{newLine}{item["content"]}{pEnd}{newLine}'
        prompt = f"{prompt}{buf}"

    item = messages[-1]
    buf = f'{pStart}{item["role"]}{newLine}{item["content"]}{newLine}'
    prompt = f"{prompt}{buf}"
    prompt += pEnd + "\n" + pStart + "assistant\n"
    return prompt


def construct_langchain_qa_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    train_prompt_template: str,
    test_prompt_template: str = None,
):
    def preprocess_qa_examples(example):
        return {
            "context": example["context"],
            "question": example["question"],
            "answer": (
                example["answers"]["text"][0]
                if example["answers"]["text"] != []
                else "unanswerable"
            ),
        }

    if test_prompt_template is None:
        test_prompt_template = train_prompt_template
    example_prompt = PromptTemplate(
        input_variables=["context", "question", "answer"],
        template=train_prompt_template,
    )
    if len(train_examples) != 0:
        train_examples = list(map(preprocess_qa_examples, train_examples))
        prompt = FewShotPromptTemplate(
            examples=train_examples,
            example_prompt=example_prompt,
            suffix=test_prompt_template.replace("{answer}", ""),
            input_variables=["context", "question"],
        )
    else:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=test_prompt_template.replace("{answer}", ""),
        )

    return prompt


def construct_translation_prompt(
    sentence: str,
    examples: List[Dict[str, str]],
    instruction: str,
    substrate_prompt: bool = False,
):
    # style: chat_prompt
    prompt = [{"role": "system", "content": instruction}]
    for example in examples:
        prompt.append({"role": "user", "content": example["source"]})
        prompt.append({"role": "assistant", "content": example["target"]})
    prompt.append({"role": "user", "content": sentence})
    return prompt if not substrate_prompt else get_substrate_prompt(prompt)


def construct_qa_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: str,
    test_prompt_template: str = None,
    chat_prompt: bool = False,
    instruction: str = "",
    substrate_prompt: bool = False,
):
    def fill_template(template, example, fill_answer=True):
        if fill_answer:
            answer = (
                "unanswerable"
                if example["answers"]["text"] == []
                else (
                    example["answers"]["text"][0]
                    if type(example["answers"]["text"]) == list
                    else example["answers"]["text"]
                )
            )
            return (
                template.replace("{context}", str(example["context"]))
                .replace("{question}", example["question"])
                .replace("{answer}", answer)
            )
        else:
            return (
                template.replace("{context}", example["context"])
                .replace("{question}", example["question"])
                .replace("{answer}", "")
                .strip()
            )

    if not chat_prompt:
        train_prompts = [
            fill_template(train_prompt_template, example) for example in train_examples
        ]
        test_prompt_input = fill_template(
            test_prompt_template, test_example, fill_answer=False
        )
        prompt_input = "\n\n".join(train_prompts + [test_prompt_input])
        test_prompt_label = test_example["answers"]["text"][0]

    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})

        for example in train_examples:
            prompt_input = fill_template(
                train_prompt_template, example, fill_answer=False
            )
            prompt_label = (
                "unanswerable"
                if example["answers"]["text"] == []
                else example["answers"]["text"][0]
            )
            # prompt_label = example["answers"]["text"][0]
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})

        test_prompt_input = fill_template(
            test_prompt_template, test_example, fill_answer=False
        )
        test_prompt_label = test_example["answers"]["text"][0]
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages
        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    return prompt_input, test_prompt_label


def construct_qa_nocontext_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: str,
    test_prompt_template: str = None,
    chat_prompt: bool = False,
    instruction: str = "",
    substrate_prompt: bool = False,
):

    def fill_template(template, example, fill_answer=True):
        if fill_answer:
            answer = (
                "unanswerable"
                if not example["answers"] or example["answers"] == ["unanswerable"]
                else example["answers"][0]
            )
            return template.replace("{question}", example["question"]).replace(
                "{answer}", answer
            )
        else:
            return (
                template.replace("{question}", example["question"])
                .replace("{answer}", "")
                .strip()
            )

    test_example["answers"] = [test_example["answers"].strip("']").strip("['")]
    if not chat_prompt:
        train_prompts = [
            fill_template(train_prompt_template, example) for example in train_examples
        ]
        test_prompt_input = fill_template(
            test_prompt_template, test_example, fill_answer=False
        )
        prompt_input = "\n\n".join(train_prompts + [test_prompt_input])
        test_prompt_label = (
            test_example["answers"][0] if test_example["answers"] else "unanswerable"
        )

    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})

        for example in train_examples:
            prompt_input = fill_template(
                train_prompt_template, example, fill_answer=False
            )
            prompt_label = (
                "unanswerable"
                if not example["answers"] or example["answers"] == ["unanswerable"]
                else example["answers"][0]
            )
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})

        test_prompt_input = fill_template(
            test_prompt_template, test_example, fill_answer=False
        )
        test_prompt_label = (
            test_example["answers"][0] if test_example["answers"] else "unanswerable"
        )
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages
        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    return prompt_input, test_prompt_label


def construct_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
    chat_prompt: bool = False,
    instruction: str = "",
    substrate_prompt: bool = False,
) -> Tuple[str, str]:
    """Creates the prompt using training few-shot examples and test example to evaluate

    Args:
        train_examples (List[Dict[str, Union[str,int]]]): List of few-shot examples
        test_example (Dict[str, Union[str,int]]): Test example to evaluate

    Returns:
        Tuple[str, str] : Final prompt string constructed to provide as input and the verbalized label
    """

    if not chat_prompt:
        train_prompts = [
            "\n".join(train_prompt_template.apply(train_example))
            for train_example in train_examples
        ]

        test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)
        prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"

    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})
        for example in train_examples:
            prompt_input, prompt_label = train_prompt_template.apply(example)
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})
        test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages
        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    return prompt_input, test_prompt_label


def construct_tagging_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    prompt_template: str,
    verbalizer: Dict[str, str] = {},
    delimiter: str = "_",
    chat_prompt: bool = False,
    instruction: str = "",
    substrate_prompt: bool = False,
) -> Tuple[str, str]:
    def apply_verbalizer(tagged_tokens):
        verbalized_tagged_tokens = []
        for tagged_token in tagged_tokens:
            token, tag = tagged_token.split(delimiter)
            verbalized_tagged_tokens.append(
                f"{token}{delimiter}{verbalizer.get(tag, tag)}"
            )
        return verbalized_tagged_tokens

    if not chat_prompt:
        train_prompts = [
            prompt_template.replace(
                "{context}", " ".join(train_example["tokens"])
            ).replace(
                "{tagged}", " ".join(apply_verbalizer(train_example["tagged_tokens"]))
            )
            for train_example in train_examples
        ]
        test_prompt_input = prompt_template.replace(
            "{context}", " ".join(test_example["tokens"])
        ).replace("{tagged}", "")

        prompt_input = "\n\n".join(train_prompts + [test_prompt_input])
        test_prompt_label = test_example["tags"]

    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})
        for train_example in train_examples:
            prompt_input = (
                prompt_template.replace("{context}", " ".join(train_example["tokens"]))
                .replace("{tagged}", "")
                .strip()
            )
            prompt_label = " ".join(
                apply_verbalizer(train_example["tagged_tokens"])
            ).strip()
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})

        test_prompt_input = (
            prompt_template.replace("{context}", " ".join(test_example["tokens"]))
            .replace("{tagged}", "")
            .strip()
        )
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages
        test_prompt_label = test_example["tags"]
        # print("tagged tokens",test_example["tagged_tokens"])

        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    return prompt_input, test_prompt_label


template = """{premise} Based on the previous passage, is it true that {hypothesis}? Yes or no? ||| {label}"""
verbalizer = {"entailed": "Yes", "contradiction": "No"}


def construct_cmxnli_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: str,
    test_prompt_template: str,
    verbalizer: Dict[str, str],
    chat_prompt: bool = False,
    instruction: str = "",
    substrate_prompt: bool = False,
) -> Tuple[str, str]:
    """Creates the prompt using training few-shot examples and test example to evaluate

    Args:
        train_examples (List[Dict[str, Union[str,int]]]): List of few-shot examples
        test_example (Dict[str, Union[str,int]]): Test example to evaluate

    Returns:
        Tuple[str, str] : Final prompt string constructed to provide as input and the verbalized label
    """

    def fill_xnli_template(example, template):
        premise = example["Premise"]
        hypothesis = example["Hypothesis"]
        label = example["Label"]

        filled_template = template.replace("{premise}", premise).replace(
            "{hypothesis}", hypothesis
        )

        filled_template = filled_template.replace("{label}", "").strip()

        return filled_template, verbalizer[label]

    if not chat_prompt:
        train_prompts = [
            "\n".join(fill_xnli_template(train_example, train_prompt_template))
            for train_example in train_examples
        ]
        test_prompt_input, test_prompt_label = fill_xnli_template(
            test_example, test_prompt_template
        )
        prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"

    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})
        for example in train_examples:
            prompt_input, prompt_label = fill_xnli_template(
                example, train_prompt_template
            )
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})
        test_prompt_input, test_prompt_label = fill_xnli_template(
            test_example, test_prompt_template
        )
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages
        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    return prompt_input, test_prompt_label


def construct_belebele_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: str,
    test_prompt_template: str,
    verbalizer: Dict[Any, str],
    chat_prompt: bool = False,
    instruction: str = "",
) -> Tuple[str, str]:
    def fill_belebele_template(example, template, chat_prompt=False):
        passage = example["flores_passage"]
        question = example["question"]
        option1 = example["mc_answer1"]
        option2 = example["mc_answer2"]
        option3 = example["mc_answer3"]
        option4 = example["mc_answer4"]

        label = example["correct_answer_num"]

        if not chat_prompt:
            filled_template = (
                template.replace("{instruction}", instruction)
                .replace("{passage}", passage)
                .replace("{query}", question)
                .replace("{A}", option1)
                .replace("{B}", option2)
                .replace("{C}", option3)
                .replace("{D}", option4)
            )

        else:
            filled_template = (
                template.replace("{passage}", passage)
                .replace("{query}", question)
                .replace("{A}", option1)
                .replace("{B}", option2)
                .replace("{C}", option3)
                .replace("{D}", option4)
            )

        return filled_template, verbalizer[label]

    if not chat_prompt:
        train_prompts = [
            "\n".join(fill_belebele_template(train_example, train_prompt_template))
            for train_example in train_examples
        ]
        test_prompt_input, test_prompt_label = fill_belebele_template(
            test_example, test_prompt_template
        )
        prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"

    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})
        for example in train_examples:
            prompt_input, prompt_label = fill_belebele_template(
                example, train_prompt_template
            )
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})
        test_prompt_input, test_prompt_label = fill_belebele_template(
            test_example, test_prompt_template
        )
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages

    return prompt_input, test_prompt_label


def construct_xstory_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: str,
    test_prompt_template: str,
    verbalizer: Dict[Any, str],
    chat_prompt: bool = False,
    instruction: str = "",
    substrate_prompt: bool = False,
) -> Tuple[str, str]:
    """Creates the prompt using training few-shot examples and test example to evaluate

    Args:
        train_examples (List[Dict[str, Union[str,int]]]): List of few-shot examples
        test_example (Dict[str, Union[str,int]]): Test example to evaluate

    Returns:
        Tuple[str, str] : Final prompt string constructed to provide as input and the verbalized label
    """

    def fill_xstory_template(example, template):
        input_sentence_1 = example["input_sentence_1"]
        input_sentence_2 = example["input_sentence_2"]
        input_sentence_3 = example["input_sentence_3"]
        input_sentence_4 = example["input_sentence_4"]
        sentence_quiz1 = example["sentence_quiz1"]
        sentence_quiz2 = example["sentence_quiz2"]

        label = example["answer_right_ending"]

        filled_template = (
            template.replace("{input_sentence_1}", input_sentence_1)
            .replace("{input_sentence_2}", input_sentence_2)
            .replace("{input_sentence_3}", input_sentence_3)
            .replace("{input_sentence_4}", input_sentence_4)
            .replace("{sentence_quiz1}", sentence_quiz1)
            .replace("{sentence_quiz2}", sentence_quiz2)
        )

        return filled_template, verbalizer[label]

    if not chat_prompt:
        train_prompts = [
            "\n".join(fill_xstory_template(train_example, train_prompt_template))
            for train_example in train_examples
        ]
        test_prompt_input, test_prompt_label = fill_xstory_template(
            test_example, test_prompt_template
        )
        prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"

    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})
        for example in train_examples:
            prompt_input, prompt_label = fill_xstory_template(
                example, train_prompt_template
            )
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})
        test_prompt_input, test_prompt_label = fill_xstory_template(
            test_example, test_prompt_template
        )
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages
        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    return prompt_input, test_prompt_label


def construct_cmsentiment_prompt(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: str,
    test_prompt_template: str,
    verbalizer: Dict[str, str],
    chat_prompt: bool = False,
    instruction: str = "",
    substrate_prompt: bool = False,
) -> Tuple[str, str]:
    """Creates the prompt using training few-shot examples and test example to evaluate

    Args:
    train_examples (List[Dict[str, Union[str,int]]]): List of few-shot examples
    test_example (Dict[str, Union[str,int]]): Test example to evaluate

    Returns: Tuple[str, str] : Final prompt string constructed to provide as input and the verbalized label
    """

    def fill_xnli_template(example, template):
        text = example["text"]
        label = example["label"]
        filled_template = template.replace("{text}", text)
        return filled_template, verbalizer[label]

    if not chat_prompt:
        train_prompts = [
            "\n".join(fill_xnli_template(train_example, train_prompt_template))
            for train_example in train_examples
        ]
        test_prompt_input, test_prompt_label = fill_xnli_template(
            test_example, test_prompt_template
        )
        prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"
    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})
        for example in train_examples:
            prompt_input, prompt_label = fill_xnli_template(
                example, train_prompt_template
            )
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})
        test_prompt_input, test_prompt_label = fill_xnli_template(
            test_example, test_prompt_template
        )
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages
        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    return prompt_input, test_prompt_label


def load_prompt_template(lang: str, prompt_name: str, dataset: str) -> Template:
    """Loads prompt template from promptsource

    Args:
        lang (str): Language specifying the split of xnli dataset for which prompt template is to be loaded
        prompt_name (str): Name of the prompt. Example: GPT-3 style

    Returns:
        Template
    """
    if dataset == "xnli" and lang in set(
        ["as", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "bn"]
    ):
        dataset_prompts = DatasetTemplates(f"Divyanshu/indicxnli/{lang}")
    elif dataset == "xcopa" and lang == "en":
        # For xcopa english data, we need to fetch from COPA in superglue instead
        dataset_prompts = DatasetTemplates("super_glue/copa")
    elif dataset == "xlsum":
        dataset_prompts = DatasetTemplates("csebuetnlp/xlsum", f"{lang}")
    else:
        dataset_prompts = DatasetTemplates(f"{dataset}/{lang}")
    return dataset_prompts[prompt_name]
