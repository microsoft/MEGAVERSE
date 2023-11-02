import os
from tqdm import tqdm
import pdb
import requests, uuid, json
from typing import Union, Optional, List, Dict
import copy
from datasets import Dataset, load_dataset
from mega.utils.env_utils import ( 
                                  BING_TRANSLATE_KEY, 
                                  BING_TRANSLATE_ENDPOINT,  
                                  COGNITIVE_API_ENDPOINT, 
                                  COGNITIVE_API_REGION, 
                                  COGNITIVE_API_VERSION,
                                  COGNITIVE_API_KEY
                                  )
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError
import json
 


subscription_key = BING_TRANSLATE_KEY
# Add your location, also known as region. The default is global.
# This is required if using a Cognitive Services resource.
location = "centralindia"
path = "/translate?api-version=3.0"
constructed_url = BING_TRANSLATE_ENDPOINT + path

headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}
 
def translate_with_azure(
                         texts: List[str],
                         source: str, 
                         targets: List[str] = ['en'],
                        endpoint = COGNITIVE_API_ENDPOINT,
                        region = COGNITIVE_API_REGION,
                        api_version = COGNITIVE_API_VERSION,
                        resource_key = COGNITIVE_API_KEY
              ) -> Dict[str, str]:
    
    credential = TranslatorCredential(resource_key, region)
    text_translator = TextTranslationClient(endpoint=endpoint, credential=credential)

    try:
        source_language = source
        target_languages = ['en']
        input_text_elements = [ InputTextItem(text = texts) ]
       
        response = text_translator.translate(content = input_text_elements, to = target_languages, from_parameter = source_language)
        translation = response[0] if response else None
 
        if translation:
            for translated_text in translation.translations:
                return translated_text.text
       
 
    except HttpResponseError as exception:
        print(f"Error Code: {exception.error.code}")
        print(f"Message: {exception.error.message}")
    

def translate_with_bing(text: str, src: str, dest: str) -> str:
    """Uses the bing translator to translate `text` from `src` language to `dest` language

    Args:
        text (str): Text to translate
        src (str): Source language to translate from
        dest (str): Language to translate to

    Returns:
        str: Translated text
    """
    params = {"api-version": "3.0", "from": src, "to": [dest]}
    body = [{"text": text}]

    try:
        request = requests.post(
            constructed_url, params=params, headers=headers, json=body
        )
        response = request.json()
        # pdb.set_trace()
        translation = response[0]["translations"][0]["text"]
    except:
        pdb.set_trace()
        translation = "<MT Failed>"

    return translation


def translate_xnli(
    xnli_dataset: Dataset, src: str, dest: str, save_path: Optional[str] = None
) -> Dataset:
    """Translate premise and hypothesis of xnli dataset

    Args:
        xnli_dataset (Dataset): Some split (train, test, val) of XNLI dataset
        src (str): Source language to translate from
        dest (str): Language to translate to
        save_path (str, optional): Path to store translated dataset. Doesn't store if set to None. Defaults to None.

    Returns:
        Dataset: Translated Dataset
    """

    # Translate premise
    xnli_dataset = xnli_dataset.map(
        lambda example: {"premise": translate_with_azure(example["premise"], src, dest)},
        num_proc=1,
        load_from_cache_file=False,
    )

    # Translate hypothesis
    xnli_dataset = xnli_dataset.map(
        lambda example: {
            "hypothesis": translate_with_azure(example["hypothesis"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        xnli_dataset.save_to_disk(save_path)

    return xnli_dataset


def translate_xcopa(
    xcopa_dataset: Dataset, src: str, dest: str, save_path: Optional[str] = None
) -> Dataset:
    """Translate premise, choices and questions of xcopa dataset

    Args:
        xcopa_dataset (Dataset): Some split (test, val) of XCOPA dataset
        src (str): Source language to translate from
        dest (str): Language to translate to
        save_path (str, optional): Path to store translated dataset. Doesn't store if set to None. Defaults to None.

    Returns:
        Dataset: Translated Dataset
    """

    # Translate premise
    xcopa_dataset = xcopa_dataset.map(
        lambda example: {"premise": translate_with_azure(example["premise"], src, dest)},
        num_proc=4,
        load_from_cache_file=False,
        desc="translating premise"
    )

    # Translate choice1
    xcopa_dataset = xcopa_dataset.map(
        lambda example: {
            "choice1": translate_with_azure(example["choice1"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
        desc="translating choice1"
    )
    
    # Translate choice2
    xcopa_dataset = xcopa_dataset.map(
        lambda example: {
            "choice2": translate_with_azure(example["choice2"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
        desc="translating choice2"
    )
    
    
    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        xcopa_dataset.save_to_disk(save_path)

    return xcopa_dataset

def translate_belebele(
    belebele_dataset: Dataset, src: str, dest: str, save_path: Optional[str] = None
) -> Dataset:
    """Translate passage, questions and choices of belebele dataset

    Args:
        belebele_dataset (Dataset): test split of belebele dataset
        src (str): Source language to translate from
        dest (str): Language to translate to
        save_path (str, optional): Path to store translated dataset. Doesn't store if set to None. Defaults to None.

    Returns:
        Dataset: Translated Dataset
    """
    # Translate passage
    belebele_dataset = belebele_dataset.map(
        lambda example: {
            "flores_passage": translate_with_azure(example["flores_passage"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate question
    belebele_dataset = belebele_dataset.map(
        lambda example: {
            "question": translate_with_azure(example["question"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate mc_answer1
    belebele_dataset = belebele_dataset.map(
        lambda example: {
            "mc_answer1": translate_with_azure(example["mc_answer1"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate mc_answer2
    belebele_dataset = belebele_dataset.map(
        lambda example: {
            "mc_answer2": translate_with_azure(example["mc_answer2"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )
    
    # Translate mc_answer3
    belebele_dataset = belebele_dataset.map(
        lambda example: {
            "mc_answer3": translate_with_azure(example["mc_answer3"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )
    
    # Translate mc_answer4
    belebele_dataset = belebele_dataset.map(
        lambda example: {
            "mc_answer4": translate_with_azure(example["mc_answer4"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )


    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        belebele_dataset.save_to_disk(save_path)

    return belebele_dataset



def translate_pawsx(
    pawsx_dataset: Dataset, src: str, dest: str, save_path: Optional[str] = None
) -> Dataset:
    """Translate s1 and s2 of pawsx dataset

    Args:
        pawsx_dataset (Dataset): Some split (train, test, val) of pawsx dataset
        src (str): Source language to translate from
        dest (str): Language to translate to
        save_path (str, optional): Path to store translated dataset. Doesn't store if set to None. Defaults to None.

    Returns:
        Dataset: Translated Dataset
    """

    # Translate premise
    pawsx_dataset = pawsx_dataset.map(
        lambda example: {
            "sentence1": translate_with_azure(example["sentence1"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate hypothesis
    pawsx_dataset = pawsx_dataset.map(
        lambda example: {
            "sentence2": translate_with_azure(example["sentence2"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pawsx_dataset.save_to_disk(save_path)

    return pawsx_dataset


def translate_xstory_cloze(
    xstory_cloze_dataset: Dataset, src: str, dest: str, save_path: Optional[str] = None
) -> Dataset:
    """Translate s1 and s2 of xstory_cloze dataset

    Args:
        xstory_cloze_dataset (Dataset): Some split (train, test, val) of xstory_cloze dataset
        src (str): Source language to translate from
        dest (str): Language to translate to
        save_path (str, optional): Path to store translated dataset. Doesn't store if set to None. Defaults to None.

    Returns:
        Dataset: Translated Dataset
    """

    # Translate input_sentence_1
    xstory_cloze_dataset = xstory_cloze_dataset.map(
        lambda example: {
            "input_sentence_1": translate_with_azure(
                example["input_sentence_1"], src, dest
            )
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate input_sentence_2
    xstory_cloze_dataset = xstory_cloze_dataset.map(
        lambda example: {
            "input_sentence_2": translate_with_azure(
                example["input_sentence_2"], src, dest
            )
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate input_sentence_3
    xstory_cloze_dataset = xstory_cloze_dataset.map(
        lambda example: {
            "input_sentence_3": translate_with_azure(
                example["input_sentence_3"], src, dest
            )
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate input_sentence_4
    xstory_cloze_dataset = xstory_cloze_dataset.map(
        lambda example: {
            "input_sentence_4": translate_with_azure(
                example["input_sentence_4"], src, dest
            )
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate sentence_quiz1
    xstory_cloze_dataset = xstory_cloze_dataset.map(
        lambda example: {
            "sentence_quiz1": translate_with_azure(example["sentence_quiz1"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # Translate sentence_quiz2
    xstory_cloze_dataset = xstory_cloze_dataset.map(
        lambda example: {
            "sentence_quiz2": translate_with_azure(example["sentence_quiz2"], src, dest)
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        xstory_cloze_dataset.save_to_disk(save_path)

    return xstory_cloze_dataset
