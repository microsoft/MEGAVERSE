#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    for model in "mistralai/Mistral-7B-Instruct-v0.2"
    do
        for lang in ar bg de el en es fr hi ru sw th tr ur vi zh
        do
            for k in 8
            do
                echo "Running for language $lang and prompt ${prompt_name} and k $k"
                python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model $model -e gpt4v3 --chat_prompt --temperature 0 --from_hf_hub --timeout 30 --save_dir $save_dir
            done
        done
    done
done