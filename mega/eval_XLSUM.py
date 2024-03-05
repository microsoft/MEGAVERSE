import os
from datasets import load_dataset
import sys
import time
import json
import csv
from promptsource.templates import Template, DatasetTemplates
import yaml
import random
import openai
from mega.data.data_utils import choose_few_shot_examples
from mega.models.completion_models import model_completion
from mega.prompting.prompting_utils import get_substrate_prompt
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.misc_utils import dump_predictions
from mega.utils.env_utils import load_openai_env_variables
from yaml.loader import SafeLoader
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
import wandb
import pandas as pd
from mega.utils.substrate_llm import LLMClient
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    substrate_prompt,
):
    if not chat_prompt:
        train_prompts = [
            "\n".join(train_prompt_template.apply(example)) for example in ic_examples
        ]
        test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)
        prompt_input = "\n".join(train_prompts + [test_prompt_input]) + "\n"
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
        if substrate_prompt:
            prompt_input = get_substrate_prompt(messages)

    # print(prompt_input)
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


if __name__ == "__main__":

    env_name = "melange"
    load_openai_env_variables()
    lang = sys.argv[1]
    paramaters_file = sys.argv[2]
    save_dir = sys.argv[3]
    args = read_parameters(paramaters_file)
    prompt_name = args["prompt_names"][0]

    if args["wandb_log"]:
        wandb.init(project="debug", entity="scai-msri", config=args)
        wandb.config.lang = lang
        wandb.run.name = f"{lang}"

    instruction = INSTRUCTIONS[args["instruction_identifier"]]

    args['response_logger_root'] = f"{save_dir}/XLSum/{args['model']}/"
    
    if not os.path.exists(args["response_logger_root"]):
        os.makedirs(args["response_logger_root"], exist_ok=True)
    
    
    response_logger_file = f"{args['response_logger_root']}/{lang}_predictions.csv"
    # response_logger_file = f"{args.save_dir}/{args.dataset}/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_Verbalizer_{args.verbalizer}_FewShotK_{args.few_shot_k}"

    try:
        results = pd.read_csv(response_logger_file).to_dict("records")
    except:
        results = []
    # Loading k in context examples to pass to the model

    # print(results)

    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])

    train_dataset = load_xlsum_data(lang, "train", args["dataset_frac"])
    ic_examples = choose_few_shot_examples(train_dataset, args["k"], "random")

    # Loading samples to evaluate on
    test_examples = load_xlsum_data(lang, "test", args["dataset_frac"])

    # Delimiting the test set to run prompt selection for the model
    model = args["model"]
    
    
    if "/" in model:
        model_obj = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model)
    
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
    llm_client = LLMClient()
    # max_tokens = args["max_tokens"]
    pbar = tqdm(
        enumerate(
            test_examples.select(
                range(min(args["max_prompt_selection_samples"], len(test_examples)))
            )
        )
    )
    for idx, test_example in pbar:
        if idx < len(results):
            print(f"skipping {idx}")
            continue

        prompt, label = construct_prompt(
            ic_examples,
            test_example,
            train_prompt_templates,
            test_prompt_templates,
            args["chat_prompt"],
            instruction,
            substrate_prompt=args["substrate_prompt"],
        )

        # print(prompt)
        # if (idx+1)%8==0:
        # time.sleep(args["sleep_period"])

        # try:

        pred = model_completion(
            prompt=prompt,
            model=model,
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
            run_details=run_details,
            lang=lang,
            model_obj=model_obj,
            tokenizer=tokenizer,
            run_substrate_llm_completion=args["substrate_prompt"],
        )

        # except:
        #     print("Error in completion")
        #     pred = "Error in completion"
        batched_predictions.append(pred)
        run_details["last_processed_idx"] = idx

        # dump_predictions(idx, pred, response_logger_file)
        r1, r2, rL = compute_rouge(scorer, pred, label)
        rouge1.append(r1)
        rouge2.append(r2)
        rougeL.append(rL)
        pbar.set_description(f"ROUGE-L: {np.average(rougeL)}")

        results.append(
            {
                "xlsum_uuid": f"{test_example['id']}_{lang}",
                "label": label,
                "prediction": pred,
                "ROUGE-1": int(r1[2]),
                "ROUGE-2": int(r2[2]),
                "ROUGE-L": int(rL[2]),
            }
        )

        # print(results)

        results_df = pd.DataFrame(results)
        results_df.to_csv(response_logger_file, index=False)

    results_df = pd.DataFrame(results)

    # print(results_df)

    avg_r1 = np.average(results_df["ROUGE-1"])
    avg_r2 = np.average(results_df["ROUGE-2"])
    avg_rL = np.average(results_df["ROUGE-L"])

    if args["wandb_log"]:
        wandb.log(run_details, step=idx + 1)
        wandb.log(
            {
                "avg R1": avg_r1,
                "avg R2": avg_r2,
                "avg RL": avg_rL,
            },
            step=idx + 1,
        )

    print(
        f"Average performance for the {prompt_name} in {lang} is ({avg_r1},{avg_r2},{avg_rL})"
    )
    dump_metrics(
        lang,
        avg_r1,
        avg_r2,
        avg_rL,
        args["response_logger_root"] + args["metric_logger_path"],
    )
