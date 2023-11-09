#!/bin/bash

for dataset in IN22-Gen IN22-Conv;
do 
    echo "Evaluating for ${dataset}"
    for lang in asm_Beng ben_Beng guj_Gujr hin_Deva kas_Arab kan_Knda mal_Mlym mar_Deva npi_Deva ori_Orya pan_Guru tam_Taml tel_Telu urd_Arab;
    do
        echo "Running for language ${lang}-eng_Latn"
        python -m mega.eval_in22 \
        -d $dataset \
        -s $lang \
        -t eng_Latn \
        -k 8 \
        -e gpt4v2 \
        --model "meta-llama/Llama-2-70b-chat-hf" \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 5120 \
        --seed 42

        echo "Running for language eng_Latn-${lang}"
        python -m mega.eval_in22 \
        -d $dataset \
        -s eng_Latn \
        -t $lang \
        -k 8 \
        -e gpt4v2 \
        --model "meta-llama/Llama-2-70b-chat-hf" \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 5120 \
        --seed 42
    done
done