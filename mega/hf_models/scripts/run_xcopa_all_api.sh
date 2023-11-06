#!/bin/bash
# echo "Starting Turbo  Evaluation"

# echo "Monolingual Evaluation"
# for prompt_name in "plausible_alternatives_discrete"
# do
#     for model in "meta-llama/Llama-2-70b-chat-hf"
#     do
#         for lang in et ht id it qu sw ta th tr vi zh
#             do
#                 for k in 2 4 8
#                     do
#                         echo "Running for language $lang and prompt ${prompt_name} and k $k"
#                         python -m mega.hf_models.src.eval_xcopa -p $lang -t $lang \
#                                                                 --pivot_prompt_name "${prompt_name}" \
#                                                                 --tgt_prompt_name "${prompt_name}" \
#                                                                 -k $k \
#                                                                 --model "${model}" \
#                                                                 -e melange \
#                                                                 --chat-prompt \
#                                                                 -d xcopa \
#                                                                 --timeout 30 \
#                                                                 --use_api 
#                     done 
#             done
#     done
# done

echo "Translate Test Evaluation"
for prompt_name in "plausible_alternatives_discrete"
do
    for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-chat-hf"
    do
        for lang in et ht id it qu sw ta th tr vi zh
            do
                for k in 2 4 8
                    do
                        echo "Running for language $lang and prompt ${prompt_name} and k $k"
                        python -m mega.hf_models.src.eval_xcopa -p en -t $lang \
                                                                --pivot_prompt_name "${prompt_name}" \
                                                                --tgt_prompt_name "${prompt_name}" \
                                                                -k $k \
                                                                --model "${model}" \
                                                                -e melange \
                                                                --chat-prompt \
                                                                -d xcopa \
                                                                --timeout 30 \
                                                                --use_api \
                                                                --translate-test
                    done 
            done
    done
done