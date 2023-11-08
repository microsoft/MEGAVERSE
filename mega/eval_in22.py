import sys
import os
import json
import random
import numpy as np
import pandas as pd
from mega.data.load_datasets import (
    load_in22_dataset,
    load_flores_test_dataset,
    IN22_LANG2CODES,
)
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.env_utils import load_openai_env_variables
from mega.models.completion_models import model_completion
from mega.utils.substrate_llm import LLMClient
from mega.utils.misc_utils import dump_predictions
from mega.prompting.prompting_utils import construct_translation_prompt
from mega.utils.parser import parse_args
from tqdm import tqdm


def evaluate_IN22(
    save_preds_path,
    train_dataset,
    example_dataset,
    model,
    source,
    target,
    few_shot_k=8,
    num_evals_per_sec=2,
    substrate_prompt=False,
    temperature=0,
    max_tokens=768,
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
            example_dataset, few_shot_k, selection_criteria="random"
        )

        incontext_examples = [
            {"source": val[f"sentence_{source}"], "target": val[f"sentence_{target}"]}
            for val in incontext_examples
        ]

        prompt = construct_translation_prompt(
            datapoint[f"sentence_{source}"],
            examples=incontext_examples,
            instruction=INSTRUCTIONS["in22"].format(
                source=IN22_LANG2CODES[source], target=IN22_LANG2CODES[target]
            ),
        )

        print(prompt)
        sys.exit(0)

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
            pred = "**********"
            print(
                f'Unable to make prediction due {e}.\nLanguage Pair: {source}-{target}, id: {datapoint["id"]}, prompt: {prompt}\n'
            )

        prediction = {
            "sentence": datapoint[f"sentence_{source}"],
            "reference": datapoint[f"sentence_{target}"],
            "prediction": pred,
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

    train_split = args.dataset.split("-")[1].capitalize()
    train_dataset = load_in22_dataset(split=train_split)
    example_dataset = load_flores_test_dataset(split="dev")

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.src_lang}-{args.tgt_lang}_FewShotK_{args.few_shot_k}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_preds_path = f"{out_dir}/preds.json"
    llm_client = LLMClient() if args.substrate_prompt else None

    _, preds_df = evaluate_IN22(
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
        llm_client=llm_client,
    )

    results_dict = vars(args)

    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
