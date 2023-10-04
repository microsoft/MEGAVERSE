from typing import Union, List, Dict, Tuple, Optional, Any
from promptsource.templates import Template, DatasetTemplates
import pdb
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from pprint import pprint
from fastchat.conversation import get_conv_template


role2tag = {"user": 0, "assistant": 1}

def convert_to_hf_chat_prompt(
                            messages : List[Dict[str, str]]
                        ) -> str:
    
    conv = get_conv_template("llama-2") 
    
    conv.set_system_message(messages[0]['content'])
    
    for idx, example in enumerate(messages[1:]):
        role = example['role']
        content = example["content"]
        
        conv.append_message(conv.roles[role2tag[role]], content)
    
    
    conv.append_message(conv.roles[1], None)
    final_prompt_input = conv.get_prompt()  
      
    return final_prompt_input

    
    