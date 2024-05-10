#!/bin/bash

for lang in ar bg bn de el en es et fi fr he hi hu id it ja ko lt nl pl pt ro ru sw th tr uk vi zh
do
    python -m mega.eval_tag -d panx -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat_wth_instruct --tgt_prompt_name structure_prompting_chat_wth_instruct -k 8 --model "gemini-pro" --max_tokens 100 --temperature 0 --eval_on_val --num_evals_per_sec 2  -e gpt4v2
done



# hi ja