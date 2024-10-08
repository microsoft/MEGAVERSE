import sys
import os
import json
import random
import numpy as np
import pandas as pd
import wandb
from datasets import load_dataset
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.models.hf_completion_models import (
    hf_model_completion,
    hf_model_api_completion,
)
from mega.eval.hf_eval_cls import initialise_model
from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt
from mega.prompting.prompting_utils import construct_qa_prompt
from mega.utils.parser import parse_args
from tqdm import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

TYDIQA_LANG2CODES = {
    "bengali": "bn",
    "korean": "ko",
    "swahili": "sw",
    "english": "en",
    "indonesian": "id",
    "arabic": "ar",
    "finnish": "fi",
    "telugu": "te",
    "russian": "ru",
}

langcodes2lang = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "hi": "Hindi",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin",
}


PROMPTS_DICT = {
    "answer_given_context_and_question": """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is:
    {answer}""",
    "lang_instruct_answer_given_context_and_question": """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is? Please try to answer in {language} and ensure that the answer appears as it is in the passage.
    A: {answer}""",
}


def load_qa_dataset(dataset_name, lang, split, dataset_frac=1, translate_test=False):
    if dataset_name == "indicqa":
        if split != "train":
            dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}")[split]
        else:
            dataset = load_dataset("squad")[split]
    elif dataset_name == "xquad":
        if split != "train":
            dataset = load_dataset("xquad", f"xquad.{lang}")[split]
        else:
            dataset = load_dataset("squad")[split]
    elif dataset_name == "tydiqa":
        dataset = load_dataset("tydiqa", "secondary_task")[split]
        dataset = dataset.map(
            lambda example: {"lang": TYDIQA_LANG2CODES[example["id"].split("-")[0]]}
        )
        dataset = dataset.filter(lambda example: example["lang"] == lang)
    elif dataset_name == "mlqa":
        if split == "train":
            print("No Training Data for MLQA, switching to validation!")
            split = "validation"
        if translate_test:
            dataset_name = f"mlqa-translate-test.{lang}"
        else:
            dataset_name = f"mlqa.{lang}.{lang}"

        dataset = load_dataset("mlqa", dataset_name)[split]

    else:
        raise NotImplementedError()
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def evaluate_qa_chatgpt(
    train_examples,
    test_dataset,
    prompt_template,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    instruction: str = "",
    use_api: bool = False,
    chat_prompt: bool = True,
    num_evals_per_sec: int = 2,
    temperature: int = 0,
    max_tokens: int = 40,
    log_wandb: bool = True,
):
    f1_sum = 0
    em_sum = 0
    avg_em = 0
    avg_f1 = 0

    squad_metric = load("squad")

    run_details = {"num_calls": 0}

    pbar = tqdm(enumerate(test_dataset))

    preds = []
    labels = []
    f1s, ems = [], []
    for i, test_example in pbar:
        prompt, label = construct_qa_prompt(
            train_examples,
            test_example,
            train_prompt_template=prompt_template,
            test_prompt_template=prompt_template,
            chat_prompt=True,
            instruction=instruction,
        )

        # print(prompt)

        final_prompt = convert_to_hf_chat_prompt(prompt)

        # print(final_prompt)

        if use_api:
            pred = hf_model_api_completion(
                final_prompt,
                model_name=model_name,
                tokenizer=tokenizer,
                temperature=0,
                run_details=run_details,
                num_evals_per_sec=num_evals_per_sec,
                max_new_tokens=max_tokens,
            )
        else:
            pred = hf_model_completion(
                final_prompt,
                model,
                tokenizer,
                temperature=0,
                run_details=run_details,
                num_evals_per_sec=num_evals_per_sec,
                max_new_tokens=max_tokens,
            )

        prediction = {"prediction_text": pred, "id": test_example["id"]}
        reference = {}
        reference["answers"] = test_example["answers"]
        reference["id"] = test_example["id"]
        results = squad_metric.compute(predictions=[prediction], references=[reference])
        f1_sum += results["f1"]
        em_sum += results["exact_match"]

        avg_f1 = f1_sum / (i + 1)
        avg_em = em_sum / (i + 1)
        if log_wandb:
            wandb.log({"f1": avg_f1, "em": avg_em}, step=i + 1)
            wandb.log(run_details, step=i + 1)
        pbar.set_description(f"em: {avg_em} f1: {avg_f1}. {i+1}/{len(test_dataset)}")

        preds.append(prediction)
        labels.append(reference)
        f1s.append(results["f1"])
        ems.append(results["exact_match"])

    results_df = pd.DataFrame(
        {"Label": labels, "Prediction": preds, "F1-Score": f1s, "EM": ems}
    )

    metrics = squad_metric.compute(predictions=preds, references=labels)

    return metrics, results_df


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

    train_dataset = load_qa_dataset(
        args.dataset,
        lang=args.pivot_lang,
        split="train" if not args.use_val_to_prompt else "validation",
    )
    test_dataset = load_qa_dataset(
        args.dataset,
        lang=args.tgt_lang,
        split="test",
        dataset_frac=args.test_frac,
    )

    train_examples = choose_few_shot_examples(
        train_dataset, args.few_shot_k, selection_criteria=args.few_shot_selection
    )

    prompt_template = PROMPTS_DICT[args.tgt_prompt_name]

    # Loading instruction for the task
    instruction = INSTRUCTIONS["xquad"]
    print(instruction)

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_Verbalizer_{args.verbalizer}_FewShotK_{args.few_shot_k}"
    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(args.model)

    if not args.use_api:
        model, tokenizer = initialise_model(args.model)

    else:
        model = None
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    results_file = f"{out_dir}/results.json"

    if not os.path.exists(results_file):
        metrics, preds_df = evaluate_qa_chatgpt(
            train_examples,
            test_dataset,
            prompt_template,
            model=model,
            model_name=args.model,
            tokenizer=tokenizer,
            use_api=args.use_api,
            instruction=instruction,
            chat_prompt=args.chat_prompt,
            num_evals_per_sec=args.num_evals_per_sec,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            log_wandb=True,
        )

        preds_df.to_csv(f"{out_dir}/preds.csv")
        print(metrics)
        results_dict = vars(args)
        results_dict["metrics"] = metrics
        if not args.no_save:
            with open(results_file, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"Results written in {out_dir}")

        if args.log_wandb:
            wandb.log(metrics)

    else:
        print(f"Results already exist in {out_dir}")


if __name__ == "__main__":
    wandb.init()
    main(sys.argv[1:])
