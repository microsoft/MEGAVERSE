import os
import argparse
import sys
import time
import random
import json
import wandb
import numpy as np
from mega.data.load_datasets import load_belebele_dataset, load_belebele_translate_test
from mega.data.data_utils import choose_few_shot_examples
from mega.eval.eval_cls import evaluate_model
from mega.models.completion_models import model_completion 
from mega.models.hf_completion_models  import hf_model_api_completion
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
from transformers import AutoTokenizer


PROMPT_TEMPLATES = {
    "Choose the correct answer": """Passage: {passage} \nQuestion: {question}\nReferring to the passage and the question above help me pick the correct answer of the question out of the given options: \n Option 1: {option1}\n Option 2: {option2}\n Option 3: {option3} \n Option 4: {option4} \n Correct Option:""",
    }

VERBALIZER = {"default": {'1': "Option1", 
                          '2': "Option2", 
                          '3': "Option3", 
                          '4': "Option4"}}


def evaluate(
    train_dataset: Dataset,
    test_dataset: Dataset,
    prompt_template: str,
    verbalizer: Dict[Any, str],
    model: str,
    few_shot_size: int,
    selection_criteria: str = "random",
    save_preds_path: Optional[str] = None,
    num_evals_per_sec: int = 2,
    parallel_eval: bool = False,
    num_proc: Optional[int] = None,
    log_wandb: bool = False,
    chat_prompt: bool = False,
    instruction: str = "",
    timeout: int = 0,
    use_hf_model: bool = False,
    **model_params,
) -> float:
    run_details = {"num_calls": 0}

    # train_examples = choose_few_shot_examples(
    #     train_dataset, few_shot_size, selection_criteria
    # )

    train_examples = []
    
    valid_labels = [1, 2, 3, 4]
    
    tokenizer = AutoTokenizer.from_pretrained(model)

    preds = []
    labels = []
    matches = []
    running_acc = 0
    num_matches = 0
    pbar = tqdm(test_dataset)
    for test_example in pbar:
        train_examples_i = train_examples
        label = verbalizer[test_example["correct_answer_num"]]
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
            if use_hf_model:
                if chat_prompt:
                    prompt_input = convert_to_hf_chat_prompt(prompt)
                
                print(prompt_input)
                pred = hf_model_api_completion(prompt=prompt_input,
                                                   model_name=model,
                                                   tokenizer=tokenizer,
                                                   timeout=timeout)
                 
            try:
                pred = model_completion(
                    prompt,
                    model,
                    timeout=timeout,
                    **model_params,
                )
                break
            except (openai.error.InvalidRequestError, openai.error.Timeout):
                if len(train_examples_i) == 0:
                    pred = np.random.choice(valid_labels)
                    print("Exausted Everything! Giving Random Prediction Now :(")
                    break
                train_examples_i = train_examples_i[:-1]
                print(
                    f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                )

        print("pred: ",pred)
        preds.append(pred)
        labels.append(label)
        matches.append(float(pred == label))
        num_matches += float(pred == label)
        running_acc = np.mean(matches)
        pbar.set_description(f"Accuracy: {running_acc}")
        if log_wandb:
            wandb.log({"acuracy": running_acc})
        # time.sleep(1 / num_evals_per_sec)

    accuracy = num_matches / len(preds)
    results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches})

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
            args.tgt_lang, args.pivot_lang, test_dataset, data_dir="data"
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
    accuracy = evaluate(
        train_dataset,
        test_dataset,
        prompt_template=prompt_template,
        verbalizer=verbalizer,
        model=args.model,
        few_shot_size=args.few_shot_k,
        selection_criteria=args.few_shot_selection,
        num_evals_per_sec=args.num_evals_per_sec,
        parallel_eval=args.parallel_eval,
        num_proc=args.num_proc,
        log_wandb=args.log_wandb,
        chat_prompt=args.chat_prompt,
        instruction=instruction,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        use_hf_model=args.use_hf_api
    )
    print(accuracy)
    # Store results
    results_dict = vars(args)
    results_dict["metrics"] = {"accuracy": accuracy}
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    if args.log_wandb:
        wandb.log({"accuracy": accuracy})


if __name__ == "__main__":
    main(sys.argv[1:])
