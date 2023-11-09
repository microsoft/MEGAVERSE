import os
import sys
import json
import pickle
import random
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from mega.utils.parser import parse_args
from mega.utils.substrate_llm import LLMClient
from mega.utils.env_utils import load_openai_env_variables
from mega.models.completion_models import model_completion
# TODO: Unify chat and non-chat prompts
@dataclass
class XRiSAWOZArgs:
    root_dir: str = '../xrisawoz_data/' # Path to data
    num_learning_examples: int = 4 # Number of in-context examples
    seed: int = 1618 # Seed for sampling random in-context examples
    model_name: str = 'palm' # Name to be used in `model_completion`
    valid_fname: str = '0.1_valid.json' # remove 0.1 in case running on full dataset
    language: str = 'en' # Possible languages are ['en', 'hi', 'fr', 'ko', 'zh', 'enhi']
    substrate_llm: bool = False # Substrate prompt

def load_text(fname):
    with open(fname, 'r') as f:
        return f.read()

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def postprocess(task, text):
    if task == 'rg':
        return [text]
    return text

def inf_ddict():
    return defaultdict(inf_ddict)

prompts = {
    'dst': 'prompt_dst.txt',
    'api': 'prompt_api.txt',
    'da': 'prompt_da.txt',
    'rg': 'prompt_rg.txt',
}
task_to_out = {
    'dst': 'state',
    'api': 'api',
    'da': 'actions',
    'rg': 'response'
}

def main(sys_args):
    global prompts, task_to_out
    args = parse_args(sys_args)
    load_openai_env_variables()
    
    random.seed(args.seed)
    prompts = {k: load_text(args.xrisawoz_root_dir + v) for k, v in prompts.items()}
    out_fname = f'{args.xrisawoz_root_dir}/{args.tgt_lang}.{args.model.split("/")[-1]}.num_shots={args.few_shot_k}.pkl'
    if not os.path.exists(out_fname):
        out = inf_ddict()
    else:
        with open(out_fname, 'rb') as f:
            out = pickle.load(f)
    data = load_json(f'{args.xrisawoz_root_dir}/processed/{args.tgt_lang}/{args.xrisawoz_valid_fname}')['data']
    inputs = defaultdict(lambda: defaultdict(list))
    for datum in data:
        inputs[datum['train_target']][datum['turn_id']].append(datum)
    print(inputs.keys())
    try:
        for task in inputs.keys():
            for turn_id in inputs[task]:
                # Sample from the thing and make prediction
                print(task, turn_id)
                m = len(inputs[task][turn_id])
                for i, datum in enumerate(tqdm(inputs[task][turn_id])):
                    if not isinstance(out[datum['dial_id']]['turns'][turn_id][task_to_out[task]], defaultdict):
                        continue
                    if m - 1 <= args.few_shot_k:
                        correct_turn_id = turn_id
                        while len(inputs[task][correct_turn_id]) - 1 <= args.few_shot_k:
                            correct_turn_id -= 1
                    else:
                        correct_turn_id = turn_id
                    m_dash = len(inputs[task][correct_turn_id])
                    # sample examples other than this current one
                    possible_choices = [icl_i for icl_i in range(m_dash) if icl_i != i]
                    icl_indices = random.sample(possible_choices, args.few_shot_k)
                    messages = [{
                        'role': 'system',
                        'content': prompts[task]
                    }]
                    # TODO: Extract in context learning prompts somewhere nicer
                    for j, icl_i in enumerate(icl_indices):
                        icl_datum = inputs[task][correct_turn_id][icl_i]
                        messages.append({
                            'role': 'user',
                            'content': f'Example #{j+1}\nTurn ID: "{icl_datum["turn_id"]}"\nDatabase: "{icl_datum["task"]}"\nContext: "{icl_datum["input_text"]}"\nAnswer:'
                        })
                        messages.append({
                            'role': 'assistant',
                            'content': icl_datum["output_text"]
                        })
                    messages.append({
                        'role': 'user',
                        'content': f'Target Example\nTurn ID: "{datum["turn_id"]}"\nDatabase: "{datum["task"]}"\nContext: "{datum["input_text"]}"\nAnswer:'
                    })
                    # TODO: Check if it's a chat model and use a chat prompt
                    final_prompt = '\n'.join(x['content'] for x in messages) + '\n'
                    response = model_completion(
                        final_prompt, 
                        args.model, 
                        lang=args.tgt_lang[-2:], 
                        run_substrate_llm_completion=args.substrate_prompt, 
                        llm_client=LLMClient() if args.substrate_prompt else None, 
                        max_tokens=256
                    )
                    print(datum['input_text'])
                    print(datum['output_text'])
                    print(response)
                    out[datum['dial_id']]['turns'][turn_id][task_to_out[task]] = response
                    out[datum['dial_id']]['turns'][turn_id]['prompt_'+task_to_out[task]] = final_prompt                                
                    with open(out_fname, 'wb') as f:
                        pickle.dump(out, f)
    except Exception as e:
        print(e)
        with open(out_fname, 'wb') as f:
            pickle.dump(out, f)
    with open(out_fname, 'wb') as f:
        pickle.dump(out, f)

if __name__ == '__main__':
    main(sys.argv[1:])