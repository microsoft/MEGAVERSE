#!/bin/bash

echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    for lang in ar bg de el en es fr hi ru sw th tr vi zh
    do
        for k in 8
        do
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "google/gemma-7b-it" -e gpt4v3 --chat_prompt --temperature 0 --timeout 30
        done
    done
done