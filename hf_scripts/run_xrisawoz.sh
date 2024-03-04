#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

echo "Monolingual Evaluation"

for model in "google/gemma-7b-it" "google/gemma-2b-it";
do
    for lang in en hi fr ko zh
    do
        echo "Running for lang $lang"
        python -m mega.eval_xrisawoz \
            -k 4 \
            --xrisawoz_root_dir "./xrisawoz_data/" \
            --xrisawoz_valid_fname "compressed_0.1_valid.json" \
            --seed 1618 \
            --model "$model" \
            --tgt_lang "$lang" \
            --from_hf_hub \
            --chat_prompt \
            --save_dir $save_dir
    done
done