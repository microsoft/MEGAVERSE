#!/bin/bash
echo "Starting Turbo  Evaluation"
echo "Monolingual Evaluation"
for prompt_name in "plausible_alternatives_discrete"
do
    for lang in en et id it sw th tr
    do
        k=8
        echo "Running for language $lang and prompt ${prompt_name} and k $k"
        python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "gemini-pro" -d xcopa --timeout 30 --use-val-to-prompt
    done
done