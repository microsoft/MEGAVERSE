import os
import sys
import random
import json
import wandb
import numpy as np
from mega.data.load_datasets import load_xcopa_dataset, load_xcopa_translate_test
from mega.eval.hf_eval_cls import evaluate_model
from mega.prompting.prompting_utils import load_prompt_template
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.parser import parse_args


def main(sys_args):
    args = parse_args(sys_args)

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
    if args.tgt_lang == "en":
        train_dataset = load_xcopa_dataset(args.pivot_lang, split="train")
        test_dataset = load_xcopa_dataset(
            args.tgt_lang,
            split="validation",
            dataset_frac=args.test_frac,
        )
    train_dataset = load_xcopa_dataset(
        args.pivot_lang, split="train" if not args.use_val_to_prompt else "validation"
    )
    test_dataset = load_xcopa_dataset(
        args.tgt_lang,
        split="test" if not args.eval_on_val else "validation",
        dataset_frac=args.test_frac,
    )
    # ToDO: Add Translate Test Support
    if args.translate_test:
        test_dataset = load_xcopa_translate_test(
            args.tgt_lang, args.pivot_lang, test_dataset, data_dir="./data"
        )

    # Load prompt templates for train and test datasets
    if args.same_prompt_name:
        args.pivot_prompt_name = args.tgt_prompt_name
    train_prompt_template = load_prompt_template(
        args.pivot_lang, args.pivot_prompt_name, dataset="xcopa"
    )
    test_prompt_template = load_prompt_template(
        args.tgt_lang, args.tgt_prompt_name, dataset="xcopa"
    )

    out_dir = f"{args.save_dir}/xcopa/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_FewShotK_{args.few_shot_k}"
    if args.translate_test:
        out_dir = f"{out_dir}_translate_test"
    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    instruction = INSTRUCTIONS[args.dataset]
    print(instruction)

    pred_file_path = f"{out_dir}/preds.csv"
    results_file = f"{out_dir}/results.json"

    if not os.path.exists(results_file):
        accuracy = evaluate_model(
            train_dataset,
            test_dataset,
            train_prompt_template,
            test_prompt_template,
            args.model,
            args.few_shot_k,
            args.few_shot_selection,
            chat_prompt=args.chat_prompt,
            use_api=args.use_api,
            instruction=INSTRUCTIONS.get(args.dataset, ""),
            save_preds_path=pred_file_path if not args.no_save else None,
            num_evals_per_sec=args.num_evals_per_sec,
            parallel_eval=args.parallel_eval,
            num_proc=args.num_proc,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )
        print(accuracy)
        # Store results
        results_dict = vars(args)
        results_dict["metrics"] = {"accuracy": accuracy}
        if not args.no_save:
            with open(results_file, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"Results written in {out_dir}")

        if args.log_wandb:
            wandb.log({"accuracy": accuracy})

    else:
        print(f"Results already exist in {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
