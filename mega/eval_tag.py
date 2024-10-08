import os
import sys
import json
import random
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
import wandb
import torch
from datasets import Dataset
from seqeval.metrics import f1_score
from mega.utils.misc_utils import dump_predictions
from mega.models.tag_models import get_model_pred
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.prompting.prompting_utils import construct_tagging_prompt
from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from mega.models.hf_completion_models import hf_model_completion
from mega.utils.parser import parse_args
from mega.data.load_datasets import load_tagging_dataset
from mega.utils.env_utils import load_openai_env_variables


udpos_verbalizer = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating-conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper-noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating-conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

panx_verbalizer = {
    "B-PER": "begin-person",
    "I-PER": "inside-person",
    "B-ORG": "begin-organization",
    "I-ORG": "inside-organization",
    "B-LOC": "begin-location",
    "I-LOC": "inside-location",
    "O": "non-entity",
}

verbalizers = {
    "udpos": {"literal": udpos_verbalizer, "identity": {}},
    "panx": {"literal": panx_verbalizer, "identity": {}},
}

PROMPTS_DICT = {
    "structure_prompting": """C: {context}\nT: {tagged}""",
    "structure_prompting_chat": """{context}\n{tagged}""",
    "structure_prompting_chat_wth_instruct": """Tag the following sentence: "{context}"\n{tagged}""",
}


def evaluate(
    train_dataset: Dataset,
    test_dataset: Dataset,
    prompt_template: str,
    verbalizer: Dict[str, str],
    model: str,
    lang: str,
    few_shot_size: int,
    from_hf_hub: bool = False,
    delimiter: str = "_",
    selection_criteria: str = "random",
    save_preds_path: Optional[str] = None,
    log_wandb: bool = False,
    chat_prompt: bool = False,
    instruction: str = "",
    one_shot_tag: bool = True,
    **model_params,
) -> float:

    train_examples = choose_few_shot_examples(
        train_dataset, few_shot_size, selection_criteria
    )

    valid_labels = set()
    for example in train_examples:
        valid_labels.update(example["tags"])
    valid_labels = list(valid_labels)

    preds = []
    labels = []
    f1_scores = []
    try:
        with open(save_preds_path, "r") as file:
            json_data = [json.loads(line) for line in file]
        idx_set = {obj["q_idx"] for obj in json_data}
    except:
        print("No preds file found")
        idx_set = set()

    pbar = tqdm(enumerate(test_dataset))
    total_items = len(test_dataset)
    if len(idx_set) == total_items:
        print("All items already evaluated!")
        if os.path.exists(save_preds_path):
            with open(save_preds_path, "r") as file:
                print("Loading preds from file")
                json_data = [json.loads(line) for line in file]
                labels = [obj["label"] for obj in json_data]
                prediction = [obj["prediction"] for obj in json_data]
            eval_score = f1_score(labels, prediction)

        return eval_score

    if from_hf_hub:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    for idx, test_example in pbar:
        if idx in idx_set:
            continue

        train_examples_i = train_examples
        if from_hf_hub:

            prompt_input, label = construct_tagging_prompt(
                train_examples,
                test_example,
                prompt_template,
                verbalizer,
                delimiter=delimiter,
                chat_prompt=chat_prompt,
                instruction=instruction,
            )
            if chat_prompt:
                prompt_input = convert_to_hf_chat_prompt(prompt_input, model)

            pred = hf_model_completion(
                prompts=prompt_input,
                model_obj=model_obj,
                tokenizer=tokenizer,
                **model_params,
            )
            pred = pred.split()
            if len(pred) < len(label):
                label = label[: len(pred)]
            elif len(label) < len(pred):
                pred = pred[: len(label)]
            preds.append(pred)
            labels.append(label)
            dump_predictions(idx, pred, label, save_preds_path)
            try:
                f1_scores.append(f1_score(preds, labels))
            except Exception as e:
                print(f"skipping {idx} due to error {e}")
                continue
            running_f1 = f1_scores[-1]
            pbar.set_description(f"f1-score: {running_f1}")
        else:

            while len(train_examples_i) >= 1:
                pred_dict = {}
                try:
                    pred_dict = get_model_pred(
                        train_examples_i,
                        test_example,
                        prompt_template,
                        verbalizer,
                        model,
                        lang,
                        delimiter=delimiter,
                        one_shot_tag=one_shot_tag,
                        chat_prompt=chat_prompt,
                        instruction=instruction,
                        **model_params,
                    )
                    break
                except Exception:
                    if len(train_examples_i) == 0:
                        pred_dict = {
                            "prediction": np.random.choice(
                                valid_labels, len(test_example["tags"]), replace=True
                            ).tolist(),
                            "ground_truth": test_example["tags"],
                        }
                        print("exausted everything! giving random prediction now :(")
                        break
                    train_examples_i = train_examples_i[:-1]
                    print(
                        f"unable to fit context size. reducing few-size by 1. new size: {len(train_examples_i)}"
                    )
                pred_dict["prediction"] = [
                    "".join(pred) if pred != "" else np.random.choice(valid_labels)
                    for pred in pred_dict["prediction"]
                ]
                preds.append(pred_dict["prediction"])
                labels.append(pred_dict["ground_truth"])
                dump_predictions(
                    idx,
                    pred_dict["prediction"],
                    pred_dict["ground_truth"],
                    save_preds_path,
                )
                try:
                    f1_scores.append(f1_score(preds, labels))
                except Exception as e:
                    print(f"skipping {idx} due to error {e}")
                    continue
                running_f1 = f1_scores[-1]
                pbar.set_description(f"f1-score: {running_f1}")
        if log_wandb:
            wandb.log({"f1": running_f1})

    print(save_preds_path, "path")
    if os.path.exists(save_preds_path):
        print("Here")
        with open(save_preds_path, "r") as file:
            print("Loading preds from file")
            json_data = [json.loads(line) for line in file]
            labels = [obj["label"] for obj in json_data]
            prediction = [obj["prediction"] for obj in json_data]
        eval_score = f1_score(labels, prediction)

    return eval_score


def main(sys_args):
    args = parse_args(sys_args)
    load_openai_env_variables()
    print(args, "args")

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
    train_dataset = load_tagging_dataset(
        args.dataset,
        args.pivot_lang,
        split="validation" if args.use_val_to_prompt else "train",
        xtreme_dir=args.xtreme_dir,
        delimiter=args.delimiter,
    )
    test_dataset = load_tagging_dataset(
        args.dataset,
        args.tgt_lang,
        split="validation" if args.eval_on_val else "test",
        max_examples=1000,  # args.max_examples,
        dataset_frac=args.test_frac,
        xtreme_dir=args.xtreme_dir,
        delimiter=args.delimiter,
    )

    # Loading instruction for the task
    instruction = INSTRUCTIONS[args.dataset]

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_Verbalizer_{args.verbalizer}_FewShotK_{args.few_shot_k}wthInstruction"
    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_preds_path = f"{out_dir}/preds.json"
    eval_score = evaluate(
        train_dataset,
        test_dataset,
        prompt_template=PROMPTS_DICT[args.tgt_prompt_name],
        verbalizer=verbalizers[args.dataset][args.verbalizer],
        model=args.model,
        lang=args.tgt_lang,
        few_shot_size=args.few_shot_k,
        delimiter=args.delimiter,
        selection_criteria=args.few_shot_selection,
        num_evals_per_sec=args.num_evals_per_sec,
        parallel_eval=args.parallel_eval,
        num_proc=args.num_proc,
        log_wandb=args.log_wandb,
        save_preds_path=save_preds_path,
        dataset=args.dataset,
        from_hf_hub=args.from_hf_hub,
        chat_prompt=args.chat_prompt,
        instruction=instruction,
        one_shot_tag=not args.not_one_shot_tag,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(eval_score)
    results_dict = vars(args)
    results_dict["metrics"] = {"f1-score": eval_score}
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    if args.log_wandb:
        wandb.log({"f1-score": eval_score})


if __name__ == "__main__":
    main(sys.argv[1:])
