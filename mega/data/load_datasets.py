import os
import json
import xml.etree.ElementTree as ET
import warnings
from typing import Union, Optional
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from mega.utils.translator import (
    translate_xnli,
    translate_pawsx,
    translate_xstory_cloze,
    translate_xcopa,
    translate_belebele,
)
from mega.data.data_utils import read_conll_data
from typing import List

IN22_LANG2CODES = {
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "eng_Latn": "English",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}

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

BELEBELE_LANG2CODES = {
    "english": "eng_Latn",
    "spanish": "spa_Latn",
    "japanese": "jpn_Jpan",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "portuguese": "por_Latn",
    "italian": "ita_Latn",
    "chinese_simplified": "zho_Hans",
    "dutch": "nld_Latn",
    "swedish": "swe_Latn",
    "turkish": "tur_Latn",
    "danish": "dan_Latn",
    "finnish": "fin_Latn",
    "russian": "rus_Cyrl",
    "norwegian": "nob_Latn",
    "korean": "kor_Hang",
    "chinese_traditional": "zho_Hant",
    "polish": "pol_Latn",
    "turkish": "tur_Latn",
    "hebrew": "heb_Hebr",
    "arabic": "arb_Arab",
    "czech": "ces_Latn",
    "hungarian": "hun_Latn",
    "thai": "tha_Thai",
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


def load_belebele_dataset(
    lang: str, split: str = "test", dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which xnli dataset is to be loaded
        split (str): Train test or validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """

    dataset = load_dataset("facebook/belebele", split=BELEBELE_LANG2CODES[lang])

    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_belebele_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:
    tt_dir = os.path.join(
        data_dir, "belebele", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_belebele(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def load_xnli_dataset(
    lang: str, split: str, dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which xnli dataset is to be loaded
        split (str): Train test of validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    if lang in set(
        ["as", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "bn"]
    ):  ##PJ:To add except hindi
        dataset = load_dataset("Divyanshu/indicxnli", lang)[split]
    else:
        dataset = load_dataset("xnli", lang)[split]
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_xnli_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:
    tt_dir = os.path.join(
        data_dir, "xnli", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_xnli(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def load_xcopa_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:
    tt_dir = os.path.join(
        data_dir, "xcopa", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_xcopa(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def load_pawsx_dataset(
    lang: str, split: str, dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which paws-x dataset is to be loaded
        split (str): Train test of validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    dataset = load_dataset("paws-x", lang)[split]
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_pawsx_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:
    tt_dir = os.path.join(
        data_dir, "paws-x", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_pawsx(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def load_xstory_cloze_dataset(
    lang: str, split: str, dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which paws-x dataset is to be loaded
        split (str): Train test of validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    if split == "validation":
        split = "train"
    elif split == "test":
        split = "eval"
    dataset = load_dataset("juletxara/xstory_cloze", lang)[split]
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_xstory_cloze_translate_test(
    tgt_lang: str,
    pivot_lang: str = "en",
    test_dataset: Optional[Dataset] = None,
    data_dir: str = "data",
) -> Dataset:
    tt_dir = os.path.join(
        data_dir, "xstory_cloze", "translate_test", f"{tgt_lang}_{pivot_lang}"
    )
    if not os.path.exists(f"{tt_dir}/dataset_info.json"):
        if test_dataset is None:
            raise ValueError(
                "Need to provide `test_dataset`, if translate_test dataset do not exist already"
            )
        tt_dataset = translate_xstory_cloze(
            test_dataset, tgt_lang, pivot_lang, save_path=tt_dir
        )
    else:
        tt_dataset = load_from_disk(tt_dir)

    return tt_dataset


def load_xlsum_data(lang, split, dataset_frac):
    """Loads the xlsum dataset"""
    langs = [
        "oromo",
        "french",
        "amharic",
        "arabic",
        "azerbaijani",
        "bengali",
        "burmese",
        "chinese_simplified",
        "chinese_traditional",
        "welsh",
        "english",
        "kirundi",
        "gujarati",
        "hausa",
        "hindi",
        "igbo",
        "indonesian",
        "japanese",
        "korean",
        "kyrgyz",
        "marathi",
        "spanish",
        "scottish_gaelic",
        "nepali",
        "pashto",
        "persian",
        "pidgin",
        "portuguese",
        "punjabi",
        "russian",
        "serbian_cyrillic",
        "serbian_latin",
        "sinhala",
        "somali",
        "swahili",
        "tamil",
        "telugu",
        "thai",
        "tigrinya",
        "turkish",
        "ukrainian",
        "urdu",
        "uzbek",
        "vietnamese",
        "yoruba",
    ]
    if lang in langs:
        dataset = load_dataset("csebuetnlp/xlsum", lang)[split]
    else:
        print("Language not supported.")
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def parse_copa_dataset(path, split="test"):
    tree = ET.parse(f"{path}/copa-{split}.xml")
    root = tree.getroot()
    items = root.findall("item")

    dataset = []

    for item in items:
        dataset.append(
            {
                "idx": item.get("id"),
                "question": item.get("asks-for"),
                "label": int(item.get("most-plausible-alternative")) - 1,
                "premise": item.find("p").text,
                "choice1": item.find("a1").text,
                "choice2": item.find("a2").text,
            }
        )

    return Dataset.from_list(dataset)


def load_xcopa_dataset(
    lang: str, split: str, dataset_frac: float = 1.0, copa_dir="data/copa/"
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which xnli dataset is to be loaded
        split (str): Train test or validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """

    if lang != "en" and split == "train":
        warnings.warn(
            "No Training Split for Non-English languages in XCOPA. Using Validation split!"
        )
        split = "validation"
    if lang == "en":
        if split in ["train", "validation"]:
            # For english fetch data from COPA in SuperGLUE
            dataset = load_dataset("super_glue", "copa")[split]
        else:
            dataset = parse_copa_dataset(copa_dir, split="test")
    else:
        dataset = load_dataset("xcopa", lang)[split]

    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_tagging_dataset(
    dataset: str,
    lang: str,
    split: str,
    max_examples: int = -1,
    dataset_frac: float = 1.0,
    xtreme_dir: str = "xtreme/download",
    delimiter: str = "_",
) -> Union[Dataset, DatasetDict]:
    split = "dev" if split == "validation" else split

    filename = f"{xtreme_dir}/{dataset}/{split}-{lang}.tsv"
    inputs, labels = read_conll_data(filename)

    dataset = Dataset.from_dict({"tokens": inputs, "tags": labels})
    dataset = dataset.map(
        lambda example: {
            "tagged_tokens": [
                f"{token}{delimiter}{tag}"
                for token, tag in zip(example["tokens"], example["tags"])
            ]
        }
    )
    N = len(dataset)
    if max_examples == -1:
        selector = np.arange(int(N * dataset_frac))
    else:
        selector = np.arange(min(N, max_examples))
    return dataset.select(selector)


def load_qa_dataset(dataset_name, lang, split, dataset_frac=1, translate_test=False):
    if dataset_name == "indicqa":
        if split == "train":
            dataset = load_dataset("squad")["train"]
        elif split == "validation":
            dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}")["validation"]
        else:
            warnings.warn(f"No {split} Data for IndicQA, switching to validation!")
            dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}")["validation"]
    elif dataset_name == "xquad":
        if split == "train":
            dataset = load_dataset("squad")["train"]
        elif split == "validation":
            dataset = load_dataset("xquad", f"xquad.{lang}")["validation"]
        else:
            warnings.warn(f"No {split} Data for XQuAD, switching to validation!")
            dataset = load_dataset("xquad", f"xquad.{lang}")["validation"]
    elif dataset_name == "tydiqa":
        if split != "test":
            dataset = load_dataset("tydiqa", "secondary_task")[split]
        else:
            warnings.warn(f"No {split} Data for TyDiQA, switching to validation!")
            dataset = load_dataset("tydiqa", "secondary_task")["validation"]
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
        dataset = dataset.filter(lambda example: example['lang'] == lang)

    else:
        raise NotImplementedError()
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)


def load_xlsum_dataset(
    lang: str,
    split: str,
    dataset_frac: int = 1,
    max_examples: int = -1,
    translate_test: bool = False,
):
    dataset = load_dataset("xlsum", f"xlsum.{lang}")[split]
    if max_examples != -1:
        return dataset.select(np.arange(min(len(dataset), max_examples)))
    else:
        return dataset.select(np.arange(int(len(dataset) * dataset_frac)))


def load_in22_dataset(
    split: str, max_examples: int = -1, dataset_frac: float = 1.0, seed: int = 42
):
    dataset = load_dataset(f"ai4bharat/IN22-{split}", "all", split=split.lower())
    if max_examples != -1:
        dataset = dataset.shuffle(seed=seed).select(
            np.arange(min(len(dataset), max_examples))
        )
    if dataset_frac < 1.0:
        dataset = dataset.shuffle(seed=42).select(
            np.arange(int(len(dataset) * dataset_frac))
        )
    return dataset


def load_flores_test_dataset(split: str = "dev"):
    return load_dataset("facebook/flores", "all", split=split)


def load_dataset_mega(
    dataset: str,
    lang: str,
    split: str,
    max_examples: int = -1,
    dataset_frac: float = 1.0,
    translate_test: bool = False,
    xtreme_dir: str = "xtreme/download",
    delimiter: str = "_",
) -> Union[Dataset, DatasetDict]:
    if dataset == "xnli":
        if not translate_test:
            return load_xnli_dataset(lang, split, dataset_frac)
        else:
            return load_xnli_translate_test(lang, split, dataset_frac)
    elif dataset == "pawsx":
        if not translate_test:
            return load_pawsx_dataset(lang, split, dataset_frac)
        else:
            return load_pawsx_translate_test(lang, split, dataset_frac)
    elif dataset == "xcopa":
        return load_xcopa_dataset(lang, split, dataset_frac)

    elif dataset == "xstory_cloze":
        if not translate_test:
            return load_xstory_cloze_dataset(lang, split, dataset_frac)
        else:
            return load_xstory_cloze_translate_test(lang, split, dataset_frac)

    elif dataset in ["udpos", "panx"]:
        return load_tagging_dataset(
            dataset, lang, split, max_examples, dataset_frac, xtreme_dir, delimiter
        )

    elif dataset in ["indicqa", "xquad", "tydiqa", "mlqa", "afriqa"]:
        return load_qa_dataset(dataset, lang, split, dataset_frac, translate_test)

    elif dataset == "xlsum":
        return load_xlsum_dataset(
            lang, split, dataset_frac, max_examples, translate_test
        )


def load_json_datasets(dataset: str, lang: str, split: str):
    """
    Loads the dataset stored in json format
    """

    def read_json_data(filename):
        with open(filename) as f:
            data = json.load(f)
        return data

    if dataset == "xcopa":
        if split in ["train", "validation"]:
            # For english fetch data from COPA in SuperGLUE
            dataset = read_json_data(f"data/xcopa/data/{lang}/val.{lang}.jsonl")
        else:
            dataset = read_json_data(f"data/xcopa/data/{lang}/test.{lang}.jsonl")

    elif dataset == "xquad":
        dataset = read_json_data(f"data/xquad/xquad.{lang}.json")["data"][0][
            "paragraphs"
        ]

    return dataset


def get_dataset_splits(dataset: str):
    """
    Returns the possible splits of a dataset
    """

    if dataset == "xquad":
        return ["validation"]
    elif dataset == "xcopa":
        return ["validation", "test"]
    return ["train", "validation", "test"]
