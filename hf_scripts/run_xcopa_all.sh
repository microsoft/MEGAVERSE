#!/bin/bash
echo "Starting Turbo  Evaluation"
echo "Monolingual Evaluation"
for prompt_name in "plausible_alternatives_discrete"
do
    for lang in vi zh
    do
        k=8
        echo "Running for language $lang and prompt ${prompt_name} and k $k"
        python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model "dev-moonshot" -e gpt4v2 -d xcopa --timeout 30 --use-val-to-prompt --substrate_prompt --chat_prompt
    done
done