import sys
import os
import time
import json
import random
from typing import List
import string
import re
import unicodedata
from functools import partial
import numpy as np
import pandas as pd
import wandb
from datasets import load_dataset
from mega.data.load_datasets import load_in22_dataset, load_flores_test_dataset
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.env_utils import load_openai_env_variables
from mega.models.completion_models import model_completion
from mega.utils.substrate_llm import LLMClient
from mega.utils.misc_utils import dump_predictions
from mega.prompting.prompting_utils import construct_translation_prompt
from mega.utils.parser import parse_args
from tqdm import tqdm
from evaluate import load



def evaluate_qa_chatgpt(
    save_preds_path,
    train_dataset,
    example_dataset,
    model,
    source,
    target,
    few_shot_k,
    num_evals_per_sec=2,
    substrate_prompt=False,
    temperature=0,
    max_tokens=20,
    log_wandb=False,
    llm_client=None,
):

    run_details = {"num_calls": 0}

    pbar = tqdm(enumerate(train_dataset))
    preds = []

    try:
        with open(save_preds_path, "r") as file:
            json_data = [json.loads(line) for line in file]

        idx_set = {obj["q_idx"] for obj in json_data}
    except:
        idx_set = set()

    total_items = len(train_dataset)
    if len(idx_set) == total_items:
        print("All items already evaluated!")
        sys.exit(0)

    for i, datapoint in pbar:
        if i in idx_set:
            continue

        incontext_examples = choose_few_shot_examples(
        example_dataset, few_shot_k, selection_criteria='random'
        )
        incontext_examples = [{'source' : val[source], 'target' : val[target]} for val in incontext_examples]

        prompt = construct_translation_prompt(
            datapoint['sentence_'+source], 
            source=source,
            target=target, 
            examples=incontext_examples,
            instruction=INSTRUCTIONS['in22-gen']
        )
        
        try:
            pred = model_completion(
                prompt,
                model,
                run_substrate_llm_completion=substrate_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_output_tokens=max_tokens,
                run_details=run_details,
                num_evals_per_sec=num_evals_per_sec,
                llm_client=llm_client,
                lang=source,
            )

        except Exception as e:
            pred = 'Unanswerable'
            print(e)
            print(f'Unable to make prediction. Language: {source}, id: {datapoint["id"]}, prompt: {prompt}')

        prediction = {
            'sentence': datapoint['sentence_'+source],
            'prediction': pred,
            'ground_truth': datapoint['sentence_'+target]
        }

        dump_predictions(i, prediction, save_preds_path)
        preds.append(prediction)
        
    results_df = pd.DataFrame(prediction)
    return None, results_df


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
        args.tgt_lang = wandb.config.tgt_lang
        args.src_lang = wandb.config.src_lang
        args.few_shot_k = wandb.config.few_shot_k
        args.temperature = wandb.config.temperature
        args.num_proc = wandb.config.num_proc

    train_dataset = load_in22_dataset(
        lang=args.src_lang,
        split="train" if not args.use_val_to_prompt else "validation",
    )
    example_dataset = load_flores_test_dataset()

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.tgt_lang}/Src_Lang_{args.src_lang}_FewShotK_{args.few_shot_k}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_preds_path = f"{out_dir}/preds.json"
    reverse_save_preds_path = f"{out_dir}/reverse_preds.json"
    llm_client = LLMClient() if args.substrate_prompt else None

    metrics, preds_df = evaluate_qa_chatgpt(
        save_preds_path,
        train_dataset,
        example_dataset,
        model=args.model,
        source=args.src_lang,
        target=args.tgt_lang,
        few_shot_k=args.few_shot_k,
        num_evals_per_sec=args.num_evals_per_sec,
        substrate_prompt=args.substrate_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_wandb=True,
        llm_client=llm_client,
    )

    reverse_metrics, reverse_preds_df = evaluate_qa_chatgpt(
        reverse_save_preds_path,
        train_dataset,
        example_dataset,
        model=args.model,
        source=args.tgt_lang,
        target=args.src_lang,
        few_shot_k=args.few_shot_k,
        num_evals_per_sec=args.num_evals_per_sec,
        substrate_prompt=args.substrate_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_wandb=True,
        llm_client=llm_client,
    )

    # preds_df.to_csv(f"{out_dir}/preds.csv")
    print(metrics)
    print(reverse_metrics)
    results_dict = vars(args)
    results_dict["metrics"] = metrics
    results_dict["reverse_metrics"] = reverse_metrics
    
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    if args.log_wandb:
        wandb.log(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
