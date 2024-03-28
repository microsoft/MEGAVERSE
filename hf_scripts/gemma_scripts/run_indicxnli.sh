#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
    for model in "google/gemma-7b-it" "google/gemma-2b-it"
    do 
        for lang in as bn gu hi kn ml mr or pa ta te
        # for lang in gu
        do
            echo "Running for language $lang and prompt ${prompt_name}"
            python -m mega.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 8 --model $model -e "gpt4v2" --chat_prompt --temperature 0 --save_dir $save_dir
        done
    done
done
