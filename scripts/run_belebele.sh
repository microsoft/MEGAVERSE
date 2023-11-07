#!/bin/bash

echo "Monolingual Evaluation"
for prompt_name in "Choose the correct answer"
do
    for lang in english spanish german japanese french portuguese italian chinese_simplified dutch swedish turkish danish finnish russian norwegian korean chinese_traditional polish turkish hebrew arabic czech hungarian thai
    do
        for k in 0 
        do
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.eval_belebele -p $lang -t $lang \
            --pivot_prompt_name "${prompt_name}" \
            --tgt_prompt_name "${prompt_name}" \
            -k $k --model "dev-moonshot" \
            -e melange --temperature 0 \
            --log_wandb --timeout 30 \
            -d "belebele" --substrate_prompt
        done
    done
done

echo "Translate Test Evaluation"
for prompt_name in "Choose the correct answer"
do
    for lang in spanish german japanese french portuguese italian chinese_simplified dutch swedish turkish danish finnish russian norwegian korean chinese_traditional polish turkish hebrew arabic czech hungarian thai
    do
        for k in 0 
        do
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.eval_belebele -p "english" -t $lang \
            --pivot_prompt_name "${prompt_name}" \
            --tgt_prompt_name "${prompt_name}" \
            -k $k --model "dev-moonshot" \
            -e melange --substrate_prompt --chat_prompt --temperature 0 \
            --log_wandb --timeout 30 \
            -d "belebele" --translate-test
        done
    done
done



# gpt 3.5 turbo experiments
# echo "Monolingual Evaluation"
# for prompt_name in "Choose the correct answer"
# do
#     for lang in english spanish german japanese french portuguese italian chinese_simplified dutch swedish turkish danish finnish russian norwegian korean chinese_traditional polish turkish hebrew arabic czech hungarian thai
#     do
#         for k in 0 
#         do
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.eval_belebele -p $lang -t $lang \
#             --pivot_prompt_name "${prompt_name}" \
#             --tgt_prompt_name "${prompt_name}" \
#             -k $k --model "dev-gpt-35-turbo" \
#             -e melange --temperature 0 \
#             --log_wandb --timeout 30 \
#             -d "belebele" --substrate_prompt
#         done
#     done
# done


# # gpt 3.5 turbo experiments
# echo "Translate Test Evaluation"
# for prompt_name in "Choose the correct answer"
# do
#     for lang in english spanish german japanese french portuguese italian chinese_simplified dutch swedish turkish danish finnish russian norwegian korean chinese_traditional polish turkish hebrew arabic czech hungarian thai
#     do
#         for k in 0 
#         do
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.eval_belebele -p "en" -t $lang \
#             --pivot_prompt_name "${prompt_name}" \
#             --tgt_prompt_name "${prompt_name}" \
#             -k $k --model "dev-gpt-35-turbo" \
#             -e melange --temperature 0 \
#             --log_wandb --timeout 30 \
#             -d "belebele" --substrate_prompt \
#             --translate_test
#         done
#     done
# done


# # meta llama experiments
# echo "Monolingual Evaluation"
# for prompt_name in "Choose the correct answer"
# do
#     for lang in english spanish german japanese french portuguese italian chinese_simplified dutch swedish turkish danish finnish russian norwegian korean chinese_traditional polish turkish hebrew arabic czech hungarian thai
#     do
#         for k in 0 
#         do
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.eval_belebele -p $lang -t $lang \
#             --pivot_prompt_name "${prompt_name}" \
#             --tgt_prompt_name "${prompt_name}" \
#             -k $k --model "meta-llama/Llama-2-70b-chat-hf" \
#             -e gpt4v3 --chat_prompt --temperature 0 \
#             --log_wandb --timeout 30 \
#             -d "belebele"
#         done
#     done
# done



# # meta llama translate test experiments

# echo "Translate Test Evaluation"
# for prompt_name in "Choose the correct answer"
# do
#     for lang in english spanish german japanese french portuguese italian chinese_simplified dutch swedish turkish danish finnish russian norwegian korean chinese_traditional polish turkish hebrew arabic czech hungarian thai
#     do
#         for k in 0 
#         do
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.eval_belebele -p $lang -t $lang \
#             --pivot_prompt_name "${prompt_name}" \
#             --tgt_prompt_name "${prompt_name}" \
#             -k $k --model "meta-llama/Llama-2-70b-chat-hf" \
#             -e gpt4v3 --chat_prompt --temperature 0 \
#             --log_wandb --timeout 30 \
#             -d "belebele" --translate_test
#         done
#     done
# done


