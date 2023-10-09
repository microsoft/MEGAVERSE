#!/bin/bash

echo "Evaluating for MLQA"
for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf"
do
    for lang in ar de es hi vi zh en
    do
        echo "Running for language $lang"
        python -m mega.hf_models.src.eval_qa_gpt4 -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 4 --model "${model}" -e melange --chat-prompt --temperature 0 --num_evals_per_sec 2 --log_wandb --chat-prompt -d mlqa
    done
done