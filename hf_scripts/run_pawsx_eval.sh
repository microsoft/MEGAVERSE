#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

echo "Evaluating GPT-4"
echo "Monolingual Evaluation"
prompt_name="PAWS-ANLI GPT3"
langs=("en"  "zh")
k=8

for model in "google/gemma-7b-it" "google/gemma-2b-it":
do 
    for lang in "${langs[@]}"
    do
        echo "Running for $lang, $prompt_name, and $k few-shot examples"
        python -m mega.eval_pawsx -d "paws-x" -p "$lang" -t "$lang" --pivot_prompt_name "$prompt_name" --tgt_prompt_name "$prompt_name" -k "$k" --model $model -e gpt4v2 --save_dir $save_dir
    done
done 