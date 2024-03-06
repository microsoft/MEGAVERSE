import os
from typing import List, Dict, Union, Tuple, Optional
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from promptsource.templates import Template
from mega.models.completion_models import get_model_pred
from mega.models.hf_completion_models import get_hf_model_pred
from mega.data.data_utils import choose_few_shot_examples
from mega.utils.misc_utils import dump_predictions
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import json
import sys
import torch


def run_seq_eval(
    save_preds_path,
    train_examples: List[Dict[str, Union[str, int]]],
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    lang: str,
    tokenizer: AutoTokenizer = None,
    model_obj: AutoModelForCausalLM = None,
    num_evals_per_sec: int = 2,
    chat_prompt: bool = False,
    instruction: str = "",
    log_wandb: bool = False,
    substrate_prompt=False,
    timeout: int = 0,
    **model_params,
) -> Tuple[float, pd.DataFrame]:
    """Runs sequential evaluation. This is slower but would be better when limited API hits are available

    Args:
        train_examples (List[Dict[str, Union[str, int]]]): _description_
        test_dataset (Dataset): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_
        save_preds_path (Optional[str], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    preds = []
    labels = []
    matches = []
    running_acc = 0
    num_matches = 0
    valid_labels = test_prompt_template.answer_choices.split("|||")
    valid_labels = [label.strip().split()[0] for label in valid_labels]
    
    try:
        with open(save_preds_path, "r") as file:
            json_data = [json.loads(line) for line in file]
        idx_set = {obj["q_idx"] for obj in json_data}
    except:
        print("No preds file found")
        idx_set = set()
    # # print(type(test_dataset))
    pbar = tqdm(enumerate(test_dataset))
    total_items = len(test_dataset)
    print("Len of test_set", total_items)
    if len(idx_set) == total_items:
        print("All items already evaluated!")
        sys.exit(0)

    if "/" in model:
        model_obj = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model)

    for idx, test_example in pbar:
        train_examples_i = train_examples

        if idx in idx_set:
            continue

        if model_obj is not None and tokenizer is not None:
            while len(train_examples_i) >= 0:
                try:
                    pred_dict = get_hf_model_pred(
                        train_examples_i,
                        test_example,
                        train_prompt_template,
                        test_prompt_template,
                        model_name=model,
                        lang=lang,
                        model_obj=model_obj,
                        tokenizer=tokenizer,
                        chat_prompt=chat_prompt,
                        substrate_prompt=substrate_prompt,
                        instruction=instruction,
                        timeout=timeout,
                        **model_params,
                    )
                    break
                    print(pred_dict)
                except ValueError:
                    if len(train_examples_i) == 0:
                        pred_dict = {
                            "prediction": np.random.choice(
                                valid_labels
                            ),  # Initialize with a random prediction
                            "ground_truth": test_prompt_template.apply(test_example)[1],
                        }
                        print("Exausted Everything! Giving Random Prediction Now :(")
                        break
                    train_examples_i = train_examples_i[:-1]
                    print(
                        f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                    )

        else:
            while len(train_examples_i) >= 0:
                try:
                    pred_dict = get_model_pred(
                        train_examples_i,
                        test_example,
                        train_prompt_template,
                        test_prompt_template,
                        model,
                        lang,
                        chat_prompt=chat_prompt,
                        substrate_prompt=substrate_prompt,
                        instruction=instruction,
                        timeout=timeout,
                        **model_params,
                    )
                    break
                except (openai.error.InvalidRequestError, openai.error.Timeout):
                    if len(train_examples_i) == 0:
                        pred_dict = {
                            "prediction": np.random.choice(
                                valid_labels
                            ),  # Initialize with a random prediction
                            "ground_truth": test_prompt_template.apply(test_example)[1],
                        }
                        print("Exausted Everything! Giving Random Prediction Now :(")
                        break
                    train_examples_i = train_examples_i[:-1]
                    print(
                        f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                    )

        pred = pred_dict["prediction"].split("\n")[0]
        label = pred_dict["ground_truth"]

        # print(pred)
        dump_predictions(idx, pred, label, save_preds_path)

        # if pred == "Invalid request":
        #     pdb.set_trace()
        #     continue
        # enc = tiktoken.get_encoding("cl100k_base")
        # label = enc.encode(label)
        # pred = enc.encode(pred)
        # print(f"Label: {label}")
        # print(f"Prediction: {pred}")
        # print(f"label decoded: {enc.decode(label)}")
        # print(f"pred decoded: {enc.decode(pred)}")
        # sys.exit(0)
        num_matches += float(pred == label)
        preds.append(pred)
        labels.append(label)
        matches.append(float(pred == label))
        running_acc = np.mean(matches)
        pbar.set_description(f"Accuracy: {running_acc}")
        if log_wandb:
            wandb.log({"acuracy": running_acc})
        # time.sleep(1 / num_evals_per_sec)

    accuracy = num_matches / len(preds)
    results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches})

    return accuracy, results_df


def run_parallel_eval(
    train_examples: List[Dict[str, Union[str, int]]],
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    lang: str,
    num_proc: int = 4,
    **model_params,
) -> Tuple[float, pd.DataFrame]:
    """Runs parallel evaluation.
    This should be substanially fast but should be avoided when limited API hits are available

    Args:
        train_examples (List[Dict[str, Union[str, int]]]): _description_
        test_dataset (Dataset): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_
        save_preds_path (Optional[str], optional): _description_. Defaults to None.
        num_proc (int): _description_. Defaults to 4.
    Returns:
        _type_: _description_
    """

    results_dataset = test_dataset.map(
        lambda example: get_model_pred(
            train_examples,
            example,
            train_prompt_template,
            test_prompt_template,
            model,
            lang,
            **model_params,
        ),
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    preds = results_dataset["prediction"]
    labels = results_dataset["ground_truth"]
    # matches = [float(pred == label) for (pred, label) in zip(preds, labels)]
    matches = [float(pred.startswith(label)) for (pred, label) in zip(preds, labels)]
    accuracy = sum(matches) / len(preds)
    results_df = pd.DataFrame({"Label": labels, "Prediction": preds, "Match": matches})

    return accuracy, results_df


def evaluate_model(
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model: str,
    lang: str,
    few_shot_size: int,
    selection_criteria: str = "random",
    chat_prompt: bool = False,
    instruction: str = "",
    save_preds_path: Optional[str] = None,
    num_evals_per_sec: int = 2,
    parallel_eval: bool = False,
    num_proc: Optional[int] = None,
    log_wandb: bool = False,
    timeout: int = 0,
    **model_params,
) -> float:
    """Evaluates the accuracy of the model
    Note: Currently compares the exact match between the generated answer and the verbalized label
    ToDo: Find alternatives to exact match (embeddings?)

    Args:
        train_dataset (Dataset): _description_
        test_dataset (Dataset): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_
        few_shot_size (int): _description_
        selection_criteria (int): _description_
        save_preds_path (Optional[str], optional): _description_. Defaults to None.
        parallel_eval (bool): _description_

    Returns:
        float: _description_
    """

    train_examples = choose_few_shot_examples(
        train_dataset, few_shot_size, selection_criteria
    )

    if save_preds_path is not None:
        preds_dir, _ = os.path.split(save_preds_path)
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        # results_df.to_csv(save_preds_path)

    if parallel_eval:
        num_proc = 4 if num_proc is None else num_proc
        accuracy, results_df = run_parallel_eval(
            train_examples,
            test_dataset,
            train_prompt_template,
            test_prompt_template,
            model=model,
            lang=lang,
            num_proc=num_proc,
            **model_params,
        )
    else:
        accuracy, results_df = run_seq_eval(
            save_preds_path,
            train_examples,
            test_dataset,
            train_prompt_template,
            test_prompt_template,
            model=model,
            lang=lang,
            num_evals_per_sec=num_evals_per_sec,
            chat_prompt=chat_prompt,
            instruction=instruction,
            log_wandb=log_wandb,
            timeout=timeout,
            **model_params,
        )

    return accuracy
