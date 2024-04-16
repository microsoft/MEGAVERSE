#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir


echo "Running with Short contexts"

for model in "google/gemma-7b-it" "google/gemma-2b-it"
do
    for k in 4 8
    do
        for lang in as bn gu hi kn ml mr or pa ta te
        do
            echo "Running for lang $lang and k $k"
            python -m mega.eval_qa_gptturbo -p $lang -t $lang -d indicqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" -k $k --short_contexts --from_hf_hub --save_dir $save_dir --model $model --chat_prompt
        done
    done
done

# echo "Running with Long Contexts"
# for k in 4 8
# do
#     for lang in as bn gu hi kn ml mr or pa ta te
#     do
#         echo "Running for lang $lang and k $k"
#         python -m mega.eval_qa_gptindex -p $lang -t $lang -d xquad --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" --eval_on_val -k $k
#     done
# done


