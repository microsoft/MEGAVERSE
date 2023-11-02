#!/bin/bash

echo "Monolingual Evaluation"
for prompt_name in "Choose the correct answer"
do
    for lang in english 
    do
        for k in 0 
        do
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.eval_belebele -p $lang -t $lang \
            --pivot_prompt_name "${prompt_name}" \
            --tgt_prompt_name "${prompt_name}" \
            -k $k --model "meta-llama/Llama-2-70b-chat-hf" \
            -e gpt4v3 --chat-prompt --temperature 0 \
            --log_wandb --timeout 30 --use_hf_api
        done
    done
done