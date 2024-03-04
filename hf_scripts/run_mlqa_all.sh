#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

echo "Evaluating for MLQA"
# for lang in ar de es hi vi zh en
for model in "google/gemma-7b-it" "google/gemma-2b-it"
do
    for lang in en ar de es hi vi zh
    do
        echo "Running for language $lang"
        python -m mega.eval_qa_gptturbo -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" \
        --tgt_prompt_name "answer_given_context_and_question" -k 8 --model $model -e gpt4v2 \
        --temperature 0 --num_evals_per_sec 2 -d mlqa --eval_on_val --from_hf_hub --chat_prompt --save_dir $save_dir
    done
done