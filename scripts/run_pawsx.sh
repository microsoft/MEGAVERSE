#!/bin/bash

echo "Evaluating GPT-4"
echo "Monolingual Evaluation"
prompt_name="PAWS-ANLI GPT3"
langs=("de" "en" "es" "fr" "ja" "ko" "zh")
k=8

for lang in "${langs[@]}"
do
    echo "Running for $lang, $prompt_name, and $k few-shot examples"
    python -m mega.eval_pawsx -d "paws-x" -p "$lang" -t "$lang" --pivot_prompt_name "$prompt_name" --tgt_prompt_name "$prompt_name" -k "$k" --model "gemini-pro" -e gpt4v2
done