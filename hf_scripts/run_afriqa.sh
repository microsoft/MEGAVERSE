#!/bin/bash

echo "Evaluating Turbo on AfriQA"
for lang in bem fon hau ibo kin swa twi wol yor zul
do
    echo "Running for language $lang"
    python -m mega.eval_afriqa_gptturbo -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 8 --substrate_prompt --model dev-gpt-35-turbo -e melange --chat_prompt --temperature 0 --num_evals_per_sec 2 -d afriqa 
done



#!/bin/bash

echo "Evaluating GPT-4 on AfriQA"
for lang in fon hau ibo swa zul 
do
    echo "Running for language $lang"
    python -m mega.eval_afriqa_gptturbo -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 8 --model gpt-4 -e melange --chat_prompt --temperature 0 --num_evals_per_sec 2 -d afriqa --test_frac 0.15
done