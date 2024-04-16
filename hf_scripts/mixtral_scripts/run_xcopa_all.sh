#!/bin/bash


arg=${1}

save_dir=${arg:-"results"}

echo $save_dir



# echo "Starting Gemma Evaluation"
# echo "Monolingual Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for model in "google/gemma-7b-it" "google/gemma-2b-it"
#         do 
#         for lang in et ht id it qu sw ta th tr vi zh
#         do
#             k=8
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model $model -e gpt4v2 -d xcopa --timeout 30 --use-val-to-prompt --chat_prompt --save_dir $save_dir
#         done
#     done
# done


echo "Starting Gemma Evaluation"
echo "Monolingual Evaluation"
for prompt_name in "plausible_alternatives_discrete"
do
    for model in "mistralai/Mixtral-8x7B-Instruct-v0.1"
        do 
        for lang in en sw tr
        do
            k=8
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.eval_xcopa -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k $k --model $model -e gpt4v2 -d xcopa --timeout 30 --eval_on_val --chat_prompt --save_dir $save_dir
        done
    done
done




