#!/bin/bash
set -e
set -x
echo "Monolingual Evaluation"

for model in dev-moonshot;
do
    for lang in en hi fr ko zh enhi
    do
        echo "Running for lang $lang"
        python -m mega.eval_xrisawoz \
            -k 4 \
            --xrisawoz_root_dir "./xrisawoz_data/" \
            --xrisawoz_valid_fname "compressed_0.1_valid.json" \
            --seed 1618 \
            --model "$model" \
            --tgt_lang "$lang" \
            --substrate_prompt
    done
done