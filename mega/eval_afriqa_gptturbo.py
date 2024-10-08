import sys
import os
import json
import random
import torch
import string
import re
import unicodedata
from functools import partial
import numpy as np
import pandas as pd
import wandb
from datasets import load_dataset
from mega.data.data_utils import choose_few_shot_examples
from mega.prompting.instructions import INSTRUCTIONS
from mega.utils.env_utils import load_openai_env_variables
from mega.models.completion_models import (
    model_completion,
)
from mega.utils.misc_utils import dump_predictions
from mega.prompting.prompting_utils import (
    construct_qa_nocontext_prompt,
)
from mega.prompting.prompting_utils import (
    construct_qa_nocontext_prompt,
)
from mega.utils.parser import parse_args
from tqdm import tqdm
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from mega.models.hf_completion_models import (
    hf_model_api_completion,
    hf_model_completion,
)
from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt


PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)
WHITESPACE_LANGS = ["en", "es", "hi", "vi", "de", "ar"]
MIXED_SEGMENTATION_LANGS = ["zh"]

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
    "answer_given_question_only": """
    Q: {question}
    A: {answer}""",
    "lang_instruct_answer_given_question_only": """
    Q: {question}
    Please provide the answer in {language}.
    A: {answer}""",
}


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r"[\u4e00-\u9fa5]", char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer_mlqa(lang, s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif lang == "es":
            return re.sub(r"\b(un|una|unos|unas|el|la|los|las)\b", " ", text)
        elif lang == "hi":
            return text  # Hindi does not have formal articles
        elif lang == "vi":
            return re.sub(r"\b(của|là|cái|chiếc|những)\b", " ", text)
        elif lang == "de":
            return re.sub(
                r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
                " ",
                text,
            )
        elif lang == "ar":
            return re.sub("\sال^|ال", " ", text)
        elif lang == "zh":
            return text  # Chinese does not have formal articles
        else:
            raise Exception("Unknown Language {}".format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception("Unknown Language {}".format(lang))
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(PUNCT)  # set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def load_qa_dataset(dataset_name, lang, split, dataset_frac=1, translate_test=False):
    if dataset_name == "indicqa":
        if split != "train":
            dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}")[split]
        else:
            dataset = load_dataset("squad_v2")[split]
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

    elif dataset_name == "afriqa":
        dataset = load_dataset("masakhane/afriqa", lang)[split]
        dataset = dataset.filter(lambda example: example["lang"] == lang)
        dataset = dataset.filter(lambda example: example["lang"] == lang)

    else:
        raise NotImplementedError()
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def evaluate_qa_chatgpt(
    save_preds_path,
    train_examples,
    test_dataset,
    prompt_template,
    model,
    lang,
    normalize_fn=normalize_answer,
    instruction="",
    chat_prompt=True,
    from_hf_hub=False,
    use_hf_api=False,
    num_evals_per_sec=2,
    temperature=0,
    max_tokens=20,
    log_wandb=False,
    metric="squad",
    llm_client=None,
    timeout=10,
):
    f1_sum = 0
    em_sum = 0
    avg_em = 0
    avg_f1 = 0
    squad_metric = load(metric)

    run_details = {"num_calls": 0}

    pbar = tqdm(enumerate(test_dataset))
    preds = []
    labels = []
    f1s, ems = [], []
    try:
        with open(save_preds_path, "r") as file:
            # json_data = json.load(file)
            json_data = [json.loads(line) for line in file]

        idx_set = {obj["q_idx"] for obj in json_data}
    except:
        idx_set = set()
    # pbar = tqdm(enumerate(test_dataset))
    total_items = len(test_dataset)
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
        tokenizer = AutoTokenizer.from_pretrained(model)

    for i, test_example in pbar:
        test_example["id"] = str(i)
        test_example["id"] = str(i)
        # example_id = f"q_{i}"
        # if i in idx_set:
        #     continue
        train_examples_i = train_examples

        if use_hf_api or from_hf_hub:
            prompt, label = construct_qa_nocontext_prompt(
                train_examples_i,
                test_example,
                train_prompt_template=prompt_template,
                test_prompt_template=prompt_template,
                chat_prompt=chat_prompt,
                instruction=instruction
            )

            # print(prompt)
            # print(label)

            if chat_prompt:
                # print("chat prompt")
                prompt_input = convert_to_hf_chat_prompt(prompt, model)

            # print(prompt_input)

            if use_hf_api:
                pred = hf_model_api_completion(
                    prompt=prompt_input,
                    model_name=model,
                    tokenizer=tokenizer,
                    timeout=timeout,
                )

            elif from_hf_hub:
                # print("printing from hf hub")
                pred = hf_model_completion(
                    prompts=prompt_input,
                    model_name=model,
                    model_obj=model_obj,
                    tokenizer=tokenizer,
                    timeout=timeout,
                    max_new_tokens=25,
                )

        else:

            while len(train_examples_i) >= 0:
                prompt, label = construct_qa_nocontext_prompt(
                    train_examples_i,
                    test_example,
                    train_prompt_template=prompt_template,
                    test_prompt_template=prompt_template,
                    chat_prompt=chat_prompt,
                    instruction=instruction,
                )

                try:
                    pred = model_completion(
                        prompt,
                        model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_output_tokens=max_tokens,
                        run_details=run_details,
                        num_evals_per_sec=num_evals_per_sec,
                        llm_client=llm_client,
                        lang=lang,
                    )
                    break

                except Exception as e:
                    print(e)

                    if len(train_examples_i) == 0:
                        pred = ""
                        print("Exausted Everything! Giving Empty Prediction Now :(")
                        break
                    train_examples_i = train_examples_i[:-1]
                    print(
                        f"Unable To Fit Context Size. Reducing few-size by 1. New Size: {len(train_examples_i)}"
                    )

        pred = normalize_fn(pred)

        if metric == "squad":
            prediction = {"prediction_text": pred, "id": str(i)}
        else:
            no_answer_probability = float("unanswerable" in pred)
            prediction = {
                "prediction_text": pred,
                "id": str(i),
                "no_answer_probability": no_answer_probability,
            }

        reference = {
            "answers": {
                "text": test_example["answers"],
                "answer_start": [0] * len(test_example["answers"]),
            },
            "answers": {
                "text": test_example["answers"],
                "answer_start": [0] * len(test_example["answers"]),
            },
            "id": str(i),
        }

        if metric == "squad":
            results = squad_metric.compute(
                predictions=[prediction], references=[reference]
            )
        else:
            results = squad_metric.compute(
                predictions=[prediction],
                references=[reference],
                no_answer_threshold=0.9,
            )
        f1_sum += results["f1"]
        if metric == "squad":
            em_sum += results["exact_match"]
        else:
            em_sum += results["exact"]
        avg_f1 = f1_sum / (i + 1)
        avg_em = em_sum / (i + 1)
        log_wandb = False
        if log_wandb:
            wandb.log({"f1": avg_f1, "em": avg_em}, step=i + 1)
            wandb.log(run_details, step=i + 1)
        pbar.set_description(f"em: {avg_em} f1: {avg_f1}. {i+1}/{len(test_dataset)}")
        dump_predictions(i, prediction, label, save_preds_path)
        preds.append(prediction)
        labels.append(reference)
        f1s.append(results["f1"])
        if metric == "squad":
            ems.append(results["exact_match"])
        else:
            ems.append(results["exact"])
    results_df = pd.DataFrame(
        {"Label": labels, "Prediction": preds, "F1-Score": f1s, "EM": ems}
    )

    if metric == "squad":
        metrics = squad_metric.compute(predictions=preds, references=labels)
    else:
        metrics = squad_metric.compute(
            predictions=preds, references=labels, no_answer_threshold=0.9
        )

    return metrics, results_df


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

    train_dataset = load_qa_dataset(
        args.dataset,
        lang=args.pivot_lang,
        split="train" if not args.use_val_to_prompt else "validation",
    )
    test_dataset = load_qa_dataset(
        args.dataset,
        lang=args.tgt_lang,
        split="test" if not args.eval_on_val else "validation",
        dataset_frac=args.test_frac,
        translate_test=args.translate_test,
    )
    train_examples = choose_few_shot_examples(
        train_dataset, args.few_shot_k, selection_criteria=args.few_shot_selection
    )

    prompt_template = PROMPTS_DICT[args.tgt_prompt_name]

    # Loading instruction for the task
    instruction = INSTRUCTIONS["afriqa"]
    instruction = INSTRUCTIONS["afriqa"]
    print(instruction)

    out_dir = f"{args.save_dir}/{args.dataset}/{args.model}/{args.tgt_lang}/PivotLang_{args.pivot_lang}_PromptName_{args.tgt_prompt_name.replace('/','_')}_Verbalizer_{args.verbalizer}_FewShotK_{args.few_shot_k}"
    if args.translate_test:
        out_dir = f"{out_dir}_translate_test"

    if args.use_val_to_prompt:
        out_dir = f"{out_dir}_use_val_to_prompt"

    if args.dataset == "indicqa":
        out_dir += f"_with_unanswerable"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    normalize_answer_mlqa_fn = partial(
        normalize_answer_mlqa, args.tgt_lang if not args.translate_test else "en"
    )
    normalize_fn = (
        normalize_answer if args.dataset != "mlqa" else normalize_answer_mlqa_fn
    )
    save_preds_path = f"{out_dir}/preds.json"

    metrics, preds_df = evaluate_qa_chatgpt(
        save_preds_path,
        train_examples,
        test_dataset,
        prompt_template,
        model=args.model,
        instruction=instruction,
        chat_prompt=args.chat_prompt,
        num_evals_per_sec=args.num_evals_per_sec,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_wandb=True,
        use_hf_api=args.use_hf_api,
        from_hf_hub=args.from_hf_hub,
        metric="squad" if args.dataset != "indicqa" else "squad_v2",
        normalize_fn=normalize_fn,
        lang=args.tgt_lang,
        timeout=args.timeout,
    )

    print(metrics)
    results_dict = vars(args)
    results_dict["metrics"] = metrics
    if not args.no_save:
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results written in {out_dir}")

    if args.log_wandb:
        wandb.log(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
