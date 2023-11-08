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
            -k $k --model "dev-gpt-35-turbo" \
            -e melange --temperature 0 \
             --timeout 30 \
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
            -e melange --substrate_prompt --temperature 0 \
             --timeout 30 \
            -d "belebele" --translate-test
        done
    done
done



