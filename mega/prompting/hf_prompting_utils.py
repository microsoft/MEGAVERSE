from typing import Union, List, Dict, Tuple, Optional, Any
from promptsource.templates import Template, DatasetTemplates
import pdb
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from pprint import pprint

def convert_to_llama_chat_prompt(
                            messages : List[Dict[str, str]]
                        ) -> str:
    
    final_prompt_input = "<s> [INST] "
    
    # pprint(messages)
    
    final_prompt_input += f"<<SYS>> {messages[0]['content']} <</SYS>> " 
    for idx, example in enumerate(messages[1:]):
        # messages.append({"role": "user", "content": prompt_input})
        # messages.append({"role": "assistant", "content": prompt_label})
        
        # print(example)
        role = example['role']
        content = example["content"]
        
        if role == "user":
            if idx == 0:
                final_prompt_input += f"{content} [/INST]"
            else:
                final_prompt_input += f"<s> [INST] {content} [/INST] "
        
        else:
            final_prompt_input += f"{content} </s>"
        
        
    # test_prompt_input, test_prompt_label = test_prompt_template.apply(test_example)
    # # messages.append({"role": "user", "content": test_prompt_input})
    
    # final_prompt_input += f"<s>[INST] {prompt_input} [/INST]"
    
    return final_prompt_input
    
    