#!/bin/bash

for dataset in IN22-Gen IN22-Conv;
do 
    echo "Evaluating for ${dataset}"
    for lang in asm_Beng ben_Beng guj_Gujr hin_Deva kas_Arab kan_Knda mal_Mlym mar_Deva npi_Deva ori_Orya pan_Guru tam_Taml tel_Telu urd_Arab;
    do
        echo "Running for language ${lang}-eng_Latn"
        python -m mega.eval_in22 \
        -d $dataset \
        -k 8 \
        -e gpt4v2 \
        --src_trans_lang $lang \
        --tgt_trans_lang eng_Latn \
        --model dev-gpt-35-turbo \
        --substrate_prompt \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 900 \
        --seed 42

        echo "Running for language eng_Latn-${lang}"
        python -m mega.eval_in22 \
        -d $dataset \
        -k 8 \
        -e gpt4v2 \
        --src_trans_lang eng_Latn \
        --tgt_trans_lang $lang \
        --model dev-gpt-35-turbo \
        --substrate_prompt \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 900 \
        --seed 42
    done
done