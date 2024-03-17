import sys
import os
import json
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.env_utils import load_openai_env_variables
from mega.models.completion_models import model_completion
from mega.utils.substrate_llm import LLMClient
from mega.utils.misc_utils import dump_predictions
from mega.prompting.prompting_utils import construct_translation_prompt
from mega.utils.parser import parse_args
from mega.models.completion_models import model_completion
from mega.models.hf_completion_models import (
    hf_model_api_completion,
    hf_model_completion,
)
from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

from mega.data.load_datasets import (
    load_in22_dataset,
    load_flores_test_dataset,
    IN22_LANG2CODES,
)


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
    max_tokens=1024,
    llm_client=None,
    use_hf_api=False,
    from_hf_hub=False,
    chat_prompt=False,
    timeout=60,
):
    run_details = {"num_calls": 0}

    pbar = tqdm(enumerate(train_dataset), total=len(train_dataset))
    preds = []

    try:
        with open(save_preds_path, "r", encoding="utf-8") as file:
            json_data = [json.loads(line) for line in file]

        idx_set = set([obj["q_idx"] for obj in json_data])
    except:
        idx_set = set()

    total_items = len(train_dataset)
    if len(idx_set) == total_items:
        print("All items already evaluated!")
        sys.exit(0)

    if from_hf_hub:
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj.eval()

    if use_hf_api:
        model_obj = None
        tokenizer = AutoTokenizer.from_pretrained(model)

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

        instruction = INSTRUCTIONS["in22"].format(
            source=IN22_LANG2CODES[source], target=IN22_LANG2CODES[target]
        )

        prompt = construct_translation_prompt(
            datapoint[f"sentence_{source}"],
            examples=incontext_examples,
            instruction=instruction,
            substrate_prompt=substrate_prompt,
        )

        if use_hf_api or from_hf_hub:
            if chat_prompt:
                prompt_input = convert_to_hf_chat_prompt(prompt, model)

            if use_hf_api:
                pred = hf_model_api_completion(
                    prompt=prompt_input,
                    model_name=model,
                    tokenizer=tokenizer,
                    timeout=timeout,
                )

            elif from_hf_hub:
                pred = hf_model_completion(
                    prompts=prompt_input,
                    model_name=model,
                    model_obj=model_obj,
                    tokenizer=tokenizer,
                    timeout=timeout,
                    max_new_tokens=max_tokens,
                )
        else:
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
                    f'Unable to make prediction due {e}.\nLanguage Pair: {source}-{target}, id: {datapoint["id"]}'
                )

        prediction = {
            "sentence": datapoint[f"sentence_{source}"],
            "reference": datapoint[f"sentence_{target}"],
            "prediction": pred,
        }

        dump_predictions(
            i, prediction, datapoint[f"sentence_{target}"], save_preds_path
        )
        preds.append(prediction)

    return None, pd.DataFrame(preds)


def main(sys_args):
    args = parse_args(sys_args)
    load_openai_env_variables()
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_split = args.dataset.split("-")[1].capitalize()
    train_dataset = load_in22_dataset(
        split=train_split,
        max_examples=args.test_examples,
        dataset_frac=args.test_frac,
        seed=args.seed,
    )
    example_dataset = load_flores_test_dataset(split="dev")

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.src_trans_lang}-{args.tgt_trans_lang}_FewShotK_{args.few_shot_k}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_preds_path = f"{out_dir}/preds.json"
    llm_client = LLMClient() if args.substrate_prompt else None

    _, preds_df = evaluate_IN22(
        save_preds_path,
        train_dataset,
        example_dataset,
        model=args.model,
        source=args.src_trans_lang,
        target=args.tgt_trans_lang,
        few_shot_k=args.few_shot_k,
        num_evals_per_sec=args.num_evals_per_sec,
        substrate_prompt=args.substrate_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        llm_client=llm_client,
        chat_prompt=args.chat_prompt,
        timeout=args.timeout,
        use_hf_api=args.use_hf_api,
        from_hf_hub=args.from_hf_hub,
    )

    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(vars(args), fp=f, indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Results written in {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
