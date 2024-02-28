from typing import Union, List, Dict, Tuple, Optional, Any
from promptsource.templates import Template, DatasetTemplates
import pdb
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from pprint import pprint
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer
# from mega.models.hf_completion_models import MODEL2PROMPT


MODEL2PROMPT= {
    "meta-llama/Llama-2-7b-chat-hf": "llama-2",
    "meta-llama/Llama-2-13b-chat-hf": "llama-2",
    "meta-llama/Llama-2-70b-chat-hf": "llama-2",
    "google/gemma-7b-it": "gemma",
    "google/gemma-2b-it": "gemma",

}


role2tag = {"user": 0, "assistant": 1}

def convert_to_hf_chat_prompt(
                            messages : List[Dict[str, str]],
                            model: str = "llama-2",
                            # tokenizer: AutoTokenizer = None,
                        ) -> str:
    
    """
    Converts a list of messages into a chat prompt that can be used with Hugging Face conversational AI models.

    Args:
        messages (List[Dict[str, str]]): A list of messages in the form of dictionaries, where each dictionary contains a "role" key and a "content" key.
        model_class (str): The model class of the Hugging Face conversational AI model. Defaults to "llama-2".
    Returns:
        str: A chat prompt that can be used with Hugging Face conversational AI models.

    Raises:
        IndexError: If the input list is empty or if the first message in the input list does not have a "content" key.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hi, how are you?"},
        ...     {"role": "assistant", "content": "I'm doing well, thanks for asking."},
        ...     {"role": "user", "content": "Can you tell me a joke?"},
        ...     {"role": "assistant", "content": "Why did the tomato turn red? Because it saw the salad dressing!"},
        ...     {"role": "user", "content": "Haha, that's funny."},
        ...     {"role": "assistant", "content": "Glad you liked it!"},
        ... ]
        >>> convert_to_hf_chat_prompt(messages)
        'user: Hi, how are you?\nbot: I\'m doing well, thanks for asking.\nuser: Can you tell me a joke?\nbot: Why did the tomato turn red? Because it saw the salad dressing!\nuser: Haha, that\'s funny.\nbot: Glad you liked it!\nuser:\n'
    """
    
    # chat = tokenizer.apply_chat_template(messages, tokenize=False)
    
    model_class = MODEL2PROMPT[model]
    conv = get_conv_template(model_class) 
    
    conv.set_system_message(messages[0]['content'])
    
    for idx, example in enumerate(messages[1:]):
        role = example['role']
        content = example["content"]
        
        conv.append_message(conv.roles[role2tag[role]], content)
    
    
    conv.append_message(conv.roles[1], None)
    final_prompt_input = conv.get_prompt()  
    
    # print(final_prompt_input)
      
    return final_prompt_input

      
    return chat

    
    