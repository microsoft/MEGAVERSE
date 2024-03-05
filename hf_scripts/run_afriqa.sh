#!/bin/bash

# echo "Evaluating Turbo on AfriQA"
# for lang in bem fon hau ibo kin swa twi wol yor zul
# do
#     echo "Running for language $lang"
#     python -m mega.eval_afriqa_gptturbo -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 8 --substrate_prompt --model dev-gpt-35-turbo -e melange --chat_prompt --temperature 0 --num_evals_per_sec 2 -d afriqa 
# done



#!/bin/bash


arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

echo "Evaluating Gemma on AfriQA"

for model in "google/gemma-7b-it" "google/gemma-2b-it"
do 
    for lang in fon hau ibo swa zul 
    do
        echo "Running for language $lang"
        python -m mega.eval_afriqa_gptturbo -p $lang -t $lang --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k 8 --model $model -e melange --chat_prompt --temperature 0 --num_evals_per_sec 2 -d afriqa --test_frac 0.15 --from_hf_hub --save_dir $save_dir
    done
done