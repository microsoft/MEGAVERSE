from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
import torch, gc
from tqdm import tqdm
from typing import List, Dict, Union, Any
from promptsource.templates import Template
from mega.prompting.prompting_utils import construct_prompt
from mega.prompting.hf_prompting_utils import convert_to_hf_chat_prompt
from mega.data.torch_dataset import PromptDataset
from mega.hf_models.utils.variables import HF_DECODER_MODELS
from huggingface_hub import InferenceClient, AsyncInferenceClient
from huggingface_hub.inference._text_generation import OverloadedError
from mega.utils.env_utils import (
    load_openai_env_variables,
    HF_API_KEY,
    BLOOMZ_API_URL,
    HF_API_URL,
)
import time

HF_DECODER_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
]

def hf_model_api_completion(
    prompt: Union[str, List[str]],
    model_name: str,
    tokenizer: AutoTokenizer,
    **model_params,
):
    
    # print(model_name)
    
    client = InferenceClient(model=model_name, token=HF_API_KEY)
    
    
    while True:
        try:
            output = client.text_generation(prompt)
            break
        except OverloadedError:
            time.sleep(1)
        
    output = tokenizer.decode(tokenizer(output)['input_ids'], skip_special_tokens=True)
    
    return output.strip().strip("\n").strip("\r").strip("\t").strip('.')
    



def hf_model_completion(
                prompts: Union[str, List[str]],
                model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
                tokenizer: AutoTokenizer,
                timeout: int = 10000000,
                batch_size: int = 1, 
                max_new_tokens: int =2,
                **model_params,
            ):

    # torch.cuda.empty_cache()
    # gc.collect()
    
    outputs = []
    
    if isinstance(prompts, str):
        prompts = [prompts]
        
    # print(prompts[0])
    prompt_dataset = PromptDataset(prompts, tokenizer)
    
    for idx, batch in enumerate(DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False)):
        # print("entered in loop") 
        with torch.no_grad():
            # print("entered in no grad")
            try:
                # set_trace()
                output = model.generate(**batch, 
                                        max_new_tokens=max_new_tokens,
                                        return_dict_in_generate=True, 
                                        output_scores=True,
                                        min_length=20,
                                        early_stopping=False,
                                        max_time=timeout,
                                        eos_token_id=tokenizer.eos_token_id
                                        )
            except:
                output = batch
        
        # print("generation done")
        for idx, encoding in enumerate(batch['input_ids']):

            input_length = encoding.shape[0]
            generated_tokens = output.sequences[idx, input_length:]
            
            # print(generated_tokens)
            
            outputs += tokenizer.batch_decode(
                                                [generated_tokens], 
                                                skip_special_tokens=True
                                              )

    # torch.cuda.empty_cache()
    # gc.collect()
    
    return outputs[0].strip().strip("\n").strip("\r").strip("\t").strip('.')


def get_hf_model_pred(
    train_examples: List[Dict[str, Union[str, int]]],
    test_example: Dict[str, Union[str, int]],
    train_prompt_template: Template,
    test_prompt_template: Template,
    model_name: str, 
    model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM] = None,
    tokenizer: AutoTokenizer = None,
    use_api: bool = False,
    chat_prompt: bool = False,
    instruction: str = "",
    timeout: int = 0,
    **model_params,
) -> Dict[str, str]:
    """_summary_

    Args:
        train_examples (List[Dict[str, Union[str, int]]]): _description_
        test_example (Dict[str, Union[str, int]]): _description_
        train_prompt_template (Template): _description_
        test_prompt_template (Template): _description_
        model (str): _description_

    Returns:
        Dict[str, str]: _description_
    """
    
    # print(test_examples)

    # for test_example in test_examples:
    #     print(test_example)
    #     break
        
    # print(use_api)
    prompt_input, label = construct_prompt(
        train_examples,
        test_example,
        train_prompt_template,
        test_prompt_template,
        chat_prompt=chat_prompt,
        instruction=instruction,
    )
    
    # pprint(prompt_input)    
    
    
    if chat_prompt:
        prompt_input = convert_to_hf_chat_prompt(prompt_input)
    
    # print(prompt_input)
        
    if use_api:
        model_prediction = hf_model_api_completion(prompt_input, model_name, tokenizer, **model_params)
    else:
        model_prediction = hf_model_completion(
            prompt_input, model, tokenizer, timeout=timeout, **model_params
        )
    return {"prediction": model_prediction, "ground_truth": label}
