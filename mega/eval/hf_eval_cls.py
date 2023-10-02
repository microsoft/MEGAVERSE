import os
from typing import List, Dict, Union, Tuple, Optional
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from promptsource.templates import Template
from mega.models.hf_completion_models import get_hf_model_pred
from mega.models.hf_completion_models import HF_DECODER_MODELS
from mega.data.data_utils import choose_few_shot_examples
import pdb
import gc
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import GPUtil
import pprint


def initialise_model(model_name):
    torch.cuda.empty_cache()
    gc.collect()
    
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    # tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    
    deviceIDs = GPUtil.getAvailable(order = 'random', limit = 1, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])    
    
    # device_id = deviceIDs[0]
    
    device_id = 0
    
    if model_name in HF_DECODER_MODELS:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    model.config.pad_token_id = model.config.eos_token_id
    
    model.to(f"cuda:{device_id}")
    
    # model = model.to_bettertransformer()
    
    return model, tokenizer



def run_seq_eval(
    train_examples: List[Dict[str, Union[str, int]]],
    test_dataset: Dataset,
    train_prompt_template: Template,
    test_prompt_template: Template,
    model_name: str,
    num_evals_per_sec: int = 2,
    chat_prompt: bool = False,
    instruction: str = "",
    log_wandb: bool = False,
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
    
    model, tokenizer = initialise_model(model_name)
    
    pbar = tqdm(test_dataset)
    
    for test_example in pbar:
        
        # train_examples_i = train_examples

        # print(test_example)
        # while len(train_examples_i) >= 0:
        #     try:
        pred_dict = get_hf_model_pred(
            train_examples,
            test_example,
            train_prompt_template,
            test_prompt_template,
            model,
            tokenizer,
            chat_prompt=chat_prompt,
            instruction=instruction,
            timeout=timeout,
            **model_params,
        )
            #     break
            # except (openai.error.InvalidRequestError, openai.error.Timeout):
            #     if len(train_examples_i) == 0:
            #         pred_dict = {
            #             "prediction": np.random.choice(
            #                 valid_labels
            #             ),  # Initialize with a random prediction
            #             "ground_truth": test_prompt_template.apply(test_example)[1],
            #         }
            #         print("Exausted Everything! Giving Random Prediction Now :(")
            #         break
            #     train_examples_i = train_examples_i[:-1]
            #     print(
            #         f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
            #     )

        pred = pred_dict["prediction"]
        # print(pred)
        # if pred == "Invalid request":
        #     pdb.set_trace()
        #     continue
        label = pred_dict["ground_truth"]
        # print(label)
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
    model_name: str,
    chat_prompt: bool = False,
    instruction: str = "",
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

    model, tokenizer = initialise_model(model_name)
    
    results_dataset = test_dataset.map(
        lambda example: get_hf_model_pred(
            train_examples,
            example,
            train_prompt_template,
            test_prompt_template,
            model,
            tokenizer,
            chat_prompt=chat_prompt,
            instruction=instruction,
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

    if parallel_eval:
        num_proc = 4 if num_proc is None else num_proc
        accuracy, results_df = run_parallel_eval(
            train_examples,
            test_dataset,
            train_prompt_template,
            test_prompt_template,
            model_name=model,
            num_proc=num_proc,
            **model_params,
        )
    else:
        accuracy, results_df = run_seq_eval(
            train_examples,
            test_dataset,
            train_prompt_template,
            test_prompt_template,
            model_name=model,
            num_evals_per_sec=num_evals_per_sec,
            chat_prompt=chat_prompt,
            instruction=instruction,
            log_wandb=log_wandb,
            timeout=timeout,
            **model_params,
        )
    
    if save_preds_path is not None:
        preds_dir, _ = os.path.split(save_preds_path)
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        results_df.to_csv(save_preds_path)

    return accuracy
