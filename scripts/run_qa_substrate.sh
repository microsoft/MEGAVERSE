#!/bin/bash

echo "Evaluating for MLQA"
# for lang in ar de es hi vi zh en
for lang in en
do
    echo "Running for language $lang"
    python -m mega.eval_qa_gptturbo -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 8 --substrate_prompt --model dev-gpt-35-turbo -e gpt4v2 --chat_prompt --temperature 0 --num_evals_per_sec 2 -d mlqa
done