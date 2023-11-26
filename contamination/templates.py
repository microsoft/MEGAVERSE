INSTRUCTION_FOR_QUIZ_GENERATION = """
Your task is to create three variations of the given TEXT in {lang} by only replacing the words in the provided TEXT with their synonyms. The meaning and sentence structure of the three new options must exactly mirror every detail in the TEXT. You must not include the provided TEXT as an option. You must make sure that:
(1) You generate three distinct options based on the provided TEXT;
(2) Options are ordered;
(3) There is not any extra explanation and;
(4) You comply with every specific symbol and letter detail in the given TEXT;
"""


TEMPLATES = {
    "xnli": """{instruction}
--TEXT--
Premise: {premise}
Question: {hypothesis}
Label: {label} ({verbalized_label})
{format_instructions}
"""
}


VERBALIZER_XNLI = {0: "entailment", 1: "neutral", 2: "contradiction"}

INSTRUCTION_FOR_QUIZ_ANSWER = """Your task is to accurately select the option that corresponds exactly to an instance from the {dataset_split} split of the {dataset} dataset and of {lang} language. Only generate a single option letter as your answer. You must not include any extra explanation."""

TEMPLATE_FOR_QUIZ_ANSWER = """{instruction}
---
Options are listed below:
{options}
---
{format_instructions}
"""
