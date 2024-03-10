#!/bin/bash

save_dir=${1:-"results"}

echo $save_dir


for dataset in IN22-Gen IN22-Conv;
do 
    for model in "google/gemma-7b-it" "google/gemma-2b-it"
    do 
    echo "Evaluating for ${dataset} for ${model}"
    for lang in asm_Beng ben_Beng guj_Gujr hin_Deva kas_Arab kan_Knda mal_Mlym mar_Deva npi_Deva ori_Orya pan_Guru tam_Taml tel_Telu urd_Arab;
    do
        echo "Running for language ${lang}-eng_Latn"
        python -m mega.eval_in22 \
        -d $dataset \
        -k 8 \
        -e gpt4v2 \
        --src_trans_lang $lang \
        --tgt_trans_lang eng_Latn \
        --model $model \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 1024 \
        --seed 42 \
        --chat_prompt \
        --from_hf_hub \
        --save_dir $save_dir

        echo "Running for language eng_Latn-${lang}"
        python -m mega.eval_in22 \
        -d $dataset \
        -k 8 \
        -e gpt4v2 \
        --tgt_trans_lang $lang \
        --src_trans_lang eng_Latn \
        --model $model \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 1024 \
        --seed 42 \
        --chat_prompt \
        --from_hf_hub \
        --save_dir $save_dir
        done
    done
done