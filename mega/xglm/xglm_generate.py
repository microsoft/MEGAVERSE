import os
from datasets import load_dataset
import sys
import csv
from promptsource.templates import DatasetTemplates
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import yaml
import random
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.misc_utils import dump_predictions
from yaml.loader import SafeLoader
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm


def read_parameters(args_path):
    """Function to read arguments + hyperparameters for testing."""
    with open(args_path) as f:
        args = yaml.load(f, Loader=SafeLoader)
        return args


def get_key(key_path):
    with open(key_path) as f:
        key = f.read().split("\n")[0]
    return key


def load_xlsum_data(lang, split, dataset_frac):
    """Loads the xlsum dataset"""
    langs = [
        "oromo",
        "french",
        "amharic",
        "arabic",
        "azerbaijani",
        "bengali",
        "burmese",
        "chinese_simplified",
        "chinese_traditional",
        "welsh",
        "english",
        "kirundi",
        "gujarati",
        "hausa",
        "hindi",
        "igbo",
        "indonesian",
        "japanese",
        "korean",
        "kyrgyz",
        "marathi",
        "spanish",
        "scottish_gaelic",
        "nepali",
        "pashto",
        "persian",
        "pidgin",
        "portuguese",
        "punjabi",
        "russian",
        "serbian_cyrillic",
        "serbian_latin",
        "sinhala",
        "somali",
        "swahili",
        "tamil",
        "telugu",
        "thai",
        "tigrinya",
        "turkish",
        "ukrainian",
        "urdu",
        "uzbek",
        "vietnamese",
        "yoruba",
    ]
    if lang in langs:
        dataset = load_dataset("csebuetnlp/xlsum", lang)[split]
    else:
        print("Language not supported.")
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_xlsum_prompts(lang, prompt_name):
    """Loads the xlsum prompts from promptsource"""
    dataset_prompts = DatasetTemplates("csebuetnlp/xlsum", f"{lang}")
    return dataset_prompts[prompt_name]


def construct_prompt(
    ic_examples,
    test_example,
    train_prompt_template,
    test_prompt_template,
    chat_prompt,
    instruction,
):
    if not chat_prompt:
        train_prompts = [
            " ".join(train_prompt_template.apply(example)) for example in ic_examples
        ]
        test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)
        prompt_input = "\n\n".join(train_prompts + [test_prompt_input]) + "\n"
    else:
        messages = []
        if instruction != "":
            messages.append({"role": "system", "content": instruction})
        for example in ic_examples:
            prompt_input, prompt_label = train_prompt_template.apply(example)
            messages.append({"role": "user", "content": prompt_input})
            messages.append({"role": "assistant", "content": prompt_label})
        test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)
        messages.append({"role": "user", "content": test_prompt_input})
        prompt_input = messages

    return prompt_input, test_prompt_label


def dump_metrics(lang, r1, r2, rL, metric_logger_path):
    with open(metric_logger_path, "a") as f:
        csvwriter = csv.writer(f, delimiter=",")
        if not os.path.exists(metric_logger_path):
            header = ["Language", "R1", "R2", "RL"]
            csvwriter.writerow(header)
        csvwriter.writerow([f"{lang}", f"{r1}", f"{r2}", f"{rL}"])


def compute_rouge(scorer, pred, label):
    score = scorer.score(pred, label)
    return score["rouge1"], score["rouge2"], score["rougeL"]


def xglm_prediction(model, tokenizer, prompt, **model_params):
    # Encode the prompt using the tokenizer
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded_prompt["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=model_params.get("temperature", 0) != 0,
            temperature=model_params.get("temperature", 0),
            max_new_tokens=model_params.get("max_tokens", 100),
        )

    # Decode the generated text
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)


if __name__ == "__main__":
    args = read_parameters("./scripts/parameters.yaml")
    lang = sys.argv[1]
    prompt_name = args["prompt_names"][0]
    instruction = INSTRUCTIONS[args["instruction_identifier"]]

    if not os.path.exists(args["response_logger_root"]):
        os.mkdir(args["response_logger_root"])

    response_logger_file = f"{args['response_logger_root']}/{lang}_predictions.csv"
    # Loading k in context examples to pass to the model

    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])

    train_dataset = load_xlsum_data(lang, "train", args["dataset_frac"])
    ic_examples = choose_few_shot_examples(train_dataset, args["k"], "random")

    # Loading samples to evaluate on
    test_examples = load_xlsum_data(lang, "test", args["dataset_frac"])

    # Delimiting the test set to run prompt selection for the model
    model = "xglm"
    if args["prompt_selection"]:
        test_examples = load_xlsum_data(lang, "validation", args["dataset_frac"])
        model = args["turbo_identifier"]  # Used for faster inference
    else:
        print(f"Evaluation running for {lang} on Test Set of {len(test_examples)}")

    # Initializing the metric
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Load prompt templates - Note that except English and a couple of other languages prompts do not exist - so you will have to generate prompts of these languages locally.
    run_details = {
        "num_calls": 0,
        "content_filter_triggers": 0,
        "last_processed_idx": 0,
    }
    print(f"Running Evaluation for prompt: {prompt_name}")
    train_prompt_templates = load_xlsum_prompts(lang, prompt_name)
    test_prompt_templates = load_xlsum_prompts(
        lang, prompt_name
    )  #  Will ideally differ
    print(f"Evaluating for {lang} on a test set of {len(test_examples)}")
    rouge1, rouge2, rougeL, batched_predictions = [], [], [], []

    pbar = tqdm(
        enumerate(
            test_examples.select(
                range(min(args["max_prompt_selection_samples"], len(test_examples)))
            )
        )
    )

    # Initialize XGLM model
    xglm_model_name = "facebook/xglm-7.5B"
    tokenizer = AutoTokenizer.from_pretrained(xglm_model_name)
    model = AutoModelForCausalLM.from_pretrained(xglm_model_name, load_in_8bit=True)
    model.eval()

    for idx, test_example in pbar:
        prompt, label = construct_prompt(
            ic_examples,
            test_example,
            train_prompt_templates,
            test_prompt_templates,
            False,
            "",
        )
        # time.sleep(args["sleep_period"])
        pred = xglm_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
        )
        run_details["last_processed_idx"] = idx
        batched_predictions.append(pred)
        dump_predictions(idx, pred, response_logger_file)
        r1, r2, rL = compute_rouge(scorer, pred, label)
        rouge1.append(r1)
        rouge2.append(r2)
        rougeL.append(rL)
        pbar.set_description(f"ROUGE-L: {np.average(rougeL)}")

    print(
        f"Average performance for the {prompt_name} in {lang} is ({np.average(rouge1)}, {np.average(rouge2)}, {np.average(rougeL)})"
    )
    dump_metrics(
        lang,
        np.average(rouge1),
        np.average(rouge2),
        np.average(rougeL),
        args["response_logger_root"] + args["metric_logger_path"],
    )
