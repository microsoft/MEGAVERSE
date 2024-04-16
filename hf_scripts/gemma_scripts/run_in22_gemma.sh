#!/bin/bash

save_dir=${1:-"results"}

echo $save_dir


for dataset in IN22-Conv IN22-Gen;
do 
    for model in "google/gemma-2b-it" "google/gemma-7b-it" 
    do 
    echo "Evaluating for ${dataset} for ${model}"
    for lang in asm_Beng ben_Beng guj_Gujr hin_Deva kas_Arab kan_Knda mal_Mlym mar_Deva npi_Deva ory_Orya pan_Guru tam_Taml tel_Telu urd_Arab;
    do
        echo "Running ${model} for language ${lang}-eng_Latn on ${dataset}"
        python -m mega.eval_in22 \
        -d $dataset \
        -k 8 \
        -e gpt4v2 \
        --src_trans_lang $lang \
        --tgt_trans_lang eng_Latn \
        --model $model \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --max_tokens 1024 \
        --seed 42 \
        --chat_prompt \
        --from_hf_hub \
        --save_dir $save_dir

        echo "Running for ${model} for language eng_Latn-${lang} on ${dataset}"
        python -m mega.eval_in22 \
        -d $dataset \
        -k 8 \
        -e gpt4v2 \
        --tgt_trans_lang $lang \
        --src_trans_lang eng_Latn \
        --model $model \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --max_tokens 1024 \
        --seed 42 \
        --chat_prompt \
        --from_hf_hub \
        --save_dir $save_dir
        done
    done
done