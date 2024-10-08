#!/bin/bash
# for lang in ro te zh ar el et fi hi it lt pl ru tr vi bg eu fr hu ja mr pt ta uk wo
for lang in ja # ar bg de el en es et fi fr he hi hu id it ja ko lt nl pl pt ro ru th tr uk vi zh
do
    echo "Running for lang $lang"
    python -m mega.eval_tag -d udpos -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat --tgt_prompt_name structure_prompting_chat -k 8 --max_tokens 100 --temperature 0 --num_evals_per_sec 2 --model gemini-pro -e gpt4v2
done
