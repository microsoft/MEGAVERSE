#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

echo "Evaluating for XQUAD"
# for lang in ar de es hi vi zh en
for model in "mistralai/Mistral-7B-Instruct-v0.2"
do
    for lang in en ar de el es hi ro ru th tr vi zh
    do
        echo "Running for language $lang"
        python -m mega.eval_qa_gptturbo -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" \
        --tgt_prompt_name "answer_given_context_and_question" -k 8 --model $model -e gpt4v2 \
        --temperature 0 --num_evals_per_sec 2 -d xquad --eval_on_val --from_hf_hub --chat_prompt \
        --save_dir $save_dir
    done
done