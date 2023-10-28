#!/bin/bash

prompt_name="PAWS-ANLI GPT3"
# langs=("en" "de" "es" "fr" "ja" "ko" "zh")
k=8

# for lang in "${langs[@]}"
for model in "meta-llama/Llama-2-70b-chat-hf"
    do
        echo "Evaluating ${model}"
        echo "Monolingual Evaluation" 
        for lang in "en" "de" "es" "fr" "ja" "ko" "zh"
            do
                echo "Running for $lang, $prompt_name, and $k few-shot examples"
                python -m mega.hf_models.src.eval_pawsx -d "paws-x" -p "$lang" -t "$lang" --pivot_prompt_name "$prompt_name" --tgt_prompt_name "$prompt_name" -k "$k" --model "${model}" -e melange --chat-prompt --use_api
            done
    done