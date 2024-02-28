#!/bin/bash

echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    for model in "google/gemma-7b-it" "google/gemma-2b-it"
    do
        for lang in sw th tr ur vi zh
        do
            for k in 8
            do
                echo "Running for language $lang and prompt ${prompt_name} and k $k"
                python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model $model -e gpt4v3 --chat_prompt --temperature 0 --timeout 30
            done
        done
    done
done


echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    for model in "google/gemma-7b" "google/gemma-2b"
    do
        for lang in sw th tr ur vi zh
        do
            for k in 8
            do
                echo "Running for language $lang and prompt ${prompt_name} and k $k"
                python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model $model -e gpt4v3 --temperature 0 --timeout 30 
            done
        done
    done
done