#!/bin/bash

for lang in hi ja
do
    python -m mega.eval_tag -d panx -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat_wth_instruct --tgt_prompt_name structure_prompting_chat_wth_instruct -k 8 --model "palm-32k" --max_tokens 100 --temperature 0 --eval_on_val --num_evals_per_sec 2  -e gpt4v2
done



# hi ja