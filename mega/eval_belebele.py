import os
import argparse
import sys
import time
import random
import json
import wandb
import torch
import numpy as np
from mega.data.load_datasets import load_belebele_dataset, load_belebele_translate_test
from mega.data.data_utils import choose_few_shot_examples
from mega.eval.eval_cls import evaluate_model
from mega.models.completion_models import model_completion
from mega.models.hf_completion_models import (
    hf_model_api_completion,
    hf_model_completion,
)
from mega.prompting.prompting_utils import construct_belebele_prompt
from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.parser import parse_args
from mega.utils.env_utils import load_openai_env_variables
from typing import Dict, Any, Optional
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import pdb
import openai
from huggingface_hub.inference._text_generation import OverloadedError, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM
from mega.utils.substrate_llm import LLMClient


PROMPT_TEMPLATES = {
    "Choose the correct answer": "{instruction}\n###\nPassage:\n{passage}\n###\nQuery:\n{query}\n###\nChoices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n###\nAnswer:\n",
}

VERBALIZER = {"default": {"1": "A", "2": "B", "3": "C", "4": "D"}}


BELEBELE2AZURE_LANG_MAP = {
    "english": "en",
    "spanish": "es",
    "german": "de",
    "japanese": "ja",
    "french": "fr",
    "portuguese": "pt",
    "italian": "it",
    "chinese_simplified": "zh-Hans",
    "dutch": "nl",
    "swedish": "sv",
    "turkish": "tr",
    "danish": "da",
    "finnish": "fi",
    "russian": "ru",
    "norwegian": "nb",
    "korean": "ko",
    "chinese_traditional": "zh-Hant",
    "polish": "pl",
    "turkish": "tr",
    "hebrew": "he",
    "arabic": "ar",
    "czech": "cs",
    "hungarian": "hu",
    "thai": "th",
}

BELEBELE2PALM_MAP = {
    "english": "en",
    "arabic": "ar",
    "bengali": "bn",
    "bulgarian": "bg",
    "chinese_simplified": "zh",
    "chinese_traditional": "zh",
    "croation": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hebrew": "iw",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "latvian": "lv",
    "lithanian": "lt",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "ukranian": "uk",
    "vietnamese": "vi",
}


def parse_pred(pred: str) -> str:
    pred = str(pred)
    # print(pred)
    if "A" in pred or "1" in pred:
        return "A"
    elif "B" in pred or "2" in pred:
        return "B"
    elif "C" in pred or "3" in pred:
        return "C"
    elif "D" in pred or "4" in pred:
        return "D"
    else:
        return pred


def evaluate(
    train_dataset: Dataset,
    test_dataset: Dataset,
    prompt_template: str,
    verbalizer: Dict[Any, str],
    model: str,
    tokenizer: AutoTokenizer,
    model_obj: AutoModelForCausalLM,
    few_shot_size: int,
    selection_criteria: str = "random",
    save_preds_path: Optional[str] = None,
    num_evals_per_sec: int = 2,
    parallel_eval: bool = False,
    num_proc: Optional[int] = None,
    log_wandb: bool = False,
    chat_prompt: bool = False,
    model_lang: str = "en",
    instruction: str = "",
    timeout: int = 0,
    substrate_prompt: bool = False,
    use_hf_api: bool = False,
    from_hf_hub: bool = False,
    llm_client=None,
    out_dir: str = "",
    **model_params,
) -> float:
    run_details = {"num_calls": 0}

    # train_examples = choose_few_shot_examples(
    #     train_dataset, few_shot_size, selection_criteria
    # )

    train_examples = []

    valid_labels = [1, 2, 3, 4]

    if "preds.csv" in os.listdir(out_dir):
        results_df = pd.read_csv(f"{out_dir}/preds.csv")
        results = results_df.to_dict("records")

    else:
        results = []

    preds = []
    labels = []
    matches = []
    running_acc = 0
    num_matches = 0

    pred_len = len(results)
    pbar = tqdm(test_dataset)

    for idx, test_example in enumerate(pbar):
        if idx < pred_len:
            print(f"skipping {idx}")
            continue

        train_examples_i = train_examples
        label = verbalizer[test_example["correct_answer_num"]]

        if use_hf_api or from_hf_hub:
            prompt, _ = construct_belebele_prompt(
                train_examples_i,
                test_example,
                prompt_template,
                prompt_template,
                verbalizer,
                chat_prompt,
                instruction,
            )
            if chat_prompt:
                # print("chat prompt")
                prompt_input = convert_to_hf_chat_prompt(prompt, model)

            # print(prompt_input)

            if use_hf_api:
                pred = hf_model_api_completion(
                    prompt=prompt_input,
                    model_name=model,
                    tokenizer=tokenizer,
                    timeout=timeout,
                )

            elif from_hf_hub:
                # print("printing from hf hub")
                pred = hf_model_completion(
                    prompts=prompt_input,
                    model_name=model,
                    model_obj=model_obj,
                    tokenizer=tokenizer,
                    timeout=timeout,
                    max_new_tokens=25,
                )

            else:
                try:
                    while len(train_examples_i) >= 0:
                        prompt, _ = construct_belebele_prompt(
                            train_examples_i,
                            test_example,
                            prompt_template,
                            prompt_template,
                            verbalizer,
                            chat_prompt,
                            instruction,
                        )
                        pred = model_completion(
                            prompt,
                            model,
                            run_substrate_llm_completion=substrate_prompt,
                            run_details=run_details,
                            num_evals_per_sec=num_evals_per_sec,
                            llm_client=llm_client,
                            lang=model_lang,
                        )
                        break
                except (
                    openai.InvalidRequestError,
                    openai.Timeout,
                    ValidationError,
                    OverloadedError,
                ) as e:
                    if len(train_examples_i) == 0:
                        pred = np.random.choice(valid_labels)
                        print("Exausted Everything! Giving Random Prediction Now :(")
                        break
                    train_examples_i = train_examples_i[:-1]
                    print(
                        f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                    )

        pred = parse_pred(pred)
        # print(pred)
        preds.append(pred)
        labels.append(label)
        matches.append(float(pred == label))
        num_matches += float(pred == label)
        running_acc = np.mean(matches)
        pbar.set_description(f"Accuracy: {running_acc}")
        if log_wandb:
            wandb.log({"acuracy": running_acc})
        # time.sleep(1 / num_evals_per_sec)

        results.append({"Label": label, "Prediction": pred, "Match": matches[-1]})

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{out_dir}/preds.csv")

    # results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches})
    accuracy = results_df["Match"].mean()

    return accuracy, results_df


def main(sys_args):
    args = parse_args(sys_args)
    load_openai_env_variables()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project="MEGA",
            entity="msrinlp",
            config=args.__dict__,
        )

        # Need to define these for the sweep
        args.pivot_lang = wandb.config.pivot_lang
        args.tgt_lang = wandb.config.tgt_lang
        args.pivot_prompt_name = wandb.config.pivot_prompt_name
        args.tgt_prompt_name = wandb.config.tgt_prompt_name
        args.few_shot_k = wandb.config.few_shot_k
        args.temperature = wandb.config.temperature
        args.num_proc = wandb.config.num_proc

    # Load datasets for pivot and target languages
    train_dataset = load_belebele_dataset(
        args.pivot_lang, split="train" if not args.use_val_to_prompt else "validation"
    )
    test_dataset = load_belebele_dataset(
        args.tgt_lang,
        split="test" if not args.eval_on_val else "validation",
        dataset_frac=args.test_frac,
    )
    # ToDO: Add Translate Test Support
    if args.translate_test:
        test_dataset = load_belebele_translate_test(
            BELEBELE2AZURE_LANG_MAP[args.tgt_lang],
            BELEBELE2AZURE_LANG_MAP[args.pivot_lang],
            test_dataset,
            data_dir="data",
        )

    # Load prompt templates for train and test datasets
    # if args.same_prompt_name:
    #     args.pivot_prompt_name = args.tgt_prompt_name
    # train_prompt_template = construct_belebele_prompt(
    #     args.pivot_lang, args.pivot_prompt_name, dataset="belebele"
    # )
    # test_prompt_template = construct_belebele_prompt(
    #     args.tgt_lang, args.tgt_prompt_name, dataset="belebele"
    # )

    # train_examples = choose_few_shot_examples(
    #     train_dataset, args.few_shot_k, args.few_shot_selection
    # )

    train_examples = []

    out_dir = f"{args.save_dir}/belebele/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_FewShotK_{args.few_shot_k}"
    if args.translate_test:
        out_dir = f"{out_dir}_translate_test"
    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    instruction = INSTRUCTIONS[args.dataset]

    print(instruction)

    prompt_template = PROMPT_TEMPLATES[args.tgt_prompt_name]
    verbalizer = VERBALIZER["default"]

    pred_file_path = f"{out_dir}/preds.csv"

    tokenizer = None
    model_obj = None
    if args.use_hf_api or args.from_hf_hub:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.from_hf_hub:
        model_obj = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model_obj = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="flash_attention_2", device_map="auto", torch_dtype=torch.bfloat16)

    model_lang = "english" if args.translate_test else args.tgt_lang
    llm_client = LLMClient() if args.substrate_prompt else None

    accuracy, results_df = evaluate(
        train_dataset,
        test_dataset,
        prompt_template=prompt_template,
        verbalizer=verbalizer,
        model=args.model,
        tokenizer=tokenizer,
        model_obj=model_obj,
        few_shot_size=args.few_shot_k,
        selection_criteria=args.few_shot_selection,
        num_evals_per_sec=args.num_evals_per_sec,
        parallel_eval=args.parallel_eval,
        num_proc=args.num_proc,
        log_wandb=args.log_wandb,
        chat_prompt=args.chat_prompt,
        instruction=instruction,
        model_lang=(
            model_lang if args.model != "palm" else BELEBELE2PALM_MAP[model_lang]
        ),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        use_hf_api=args.use_hf_api,
        from_hf_hub=args.from_hf_hub,
        out_dir=out_dir,
        llm_client=llm_client,
        substrate_prompt=args.substrate_prompt,
    )
    print(accuracy)
    # Store results
    results_dict = vars(args)
    results_dict["metrics"] = {"accuracy": accuracy}
    print(results_dict)

    results_df.to_csv(f"{out_dir}/preds.csv")

    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    if args.log_wandb:
        wandb.log({"accuracy": accuracy})


if __name__ == "__main__":
    main(sys.argv[1:])
