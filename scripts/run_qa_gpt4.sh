#!/bin/bash

echo "Evaluating for MLQA"
for lang in ar de es hi vi zh en
do
    echo "Running for language $lang"
    python -m mega.eval_qa_gptturbo -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "dev-gpt-35-turbo" -e gpt4v2 --chat-prompt --temperature 0 --substrate_prompt --num_evals_per_sec 2 --chat-prompt -d mlqa
done