import os
from typing import List, Dict, Union, Tuple, Optional
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import Dataset
from seqeval.metrics import f1_score
from promptsource.templates import Template
from mega.models.tag_models import get_model_pred
from mega.data.data_utils import choose_few_shot_examples
from mega.utils.parser import parse_args
from mega.data.load_datasets import load_tagging_dataset
import openai
import pdb


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
    "udpos": {
        "literal": udpos_verbalizer,
        "identity": {}
    },
    "panx": {
        "literal": panx_verbalizer,
        "identity": {}
    }
}

PROMPTS_DICT = {
    "structure_prompting": """C: {context}\nT: {tagged}"""
}

def evaluate(
    train_dataset: Dataset,
    test_dataset: Dataset,
    prompt_template: str,
    verbalizer: Dict[str, str],
    model: str,
    few_shot_size: int,
    delimiter: str = "_",
    selection_criteria: str = "random",
    save_preds_path: Optional[str] = None,
    num_evals_per_sec: int = 2,
    parallel_eval: bool = False,
    num_proc: Optional[int] = None,
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
    for test_example in tqdm(test_dataset):
        train_examples_i = train_examples
        
        while len(train_examples_i) >= 1:
            try:
                pred_dict = get_model_pred(
                    train_examples_i,
                    test_example,
                    prompt_template,
                    verbalizer,
                    model,
                    delimiter=delimiter,
                    **model_params,
                )
                break
            except openai.error.InvalidRequestError:
                if len(train_examples_i) == 0:
                    pred_dict = {
                        "prediction": np.random.choice(
                            valid_labels, len(test_example["tags"]), replace=True
                        ).tolist(),
                        "ground_truth": test_example["tags"]
                    }
                    print("Exausted Everything! Giving Random Prediction Now :(")
                    break
                train_examples_i = train_examples_i[:-1]
                print(
                    f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                )
        preds.append(pred_dict["prediction"])
        labels.append(pred_dict["ground_truth"])
        f1_scores.append(f1_score(pred_dict["ground_truth"],
                                  pred_dict["prediction"]))
        time.sleep(1 / num_evals_per_sec)
        
    eval_score = f1_score(
        labels, preds
    )
    results_df = pd.DataFrame({"Label": labels,
                               "Prediction": preds,
                               "F1-Score": f1_scores})
   
    return eval_score, results_df


def main():
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
    train_dataset = load_tagging_dataset(
        args.dataset,
        args.pivot_lang,
        split="train" if not args.use_val_to_prompt else "validation",
        xtreme_dir=args.xtreme_dir,
        delimiter=args.delimiter
    )
    test_dataset = load_tagging_dataset(
        args.dataset,
        args.pivot_lang,
        split="test" if not args.eval_on_val else "validation",
        dataset_frac=args.test_frac,
        xtreme_dir=args.xtreme_dir,
        delimiter=args.delimiter
    )

    out_dir = f"{args.save_dir}/xnli/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_Verbalizer_{args.verbalizer}_FewShotK_{args.few_shot_k}"
    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    eval_score, preds_df = evaluate(
        train_dataset,
        test_dataset,
        prompt_template=PROMPTS_DICT[args.tgt_prompt_name],
        verbalizer=verbalizers[args.dataset][args.verbalizer],
        model=args.model,
        few_shot_size=args.few_shot_k,
        delimiter=args.delimiter,
        selection_criteria=args.few_shot_selection,
        num_evals_per_sec=args.num_evals_per_sec,
        parallel_eval=args.parallel_eval,
        num_proc=args.num_proc,
        temperature=args.temperature,
        top_p=args.top_p
    )
    preds_df.to_csv(out_dir)
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
    main()