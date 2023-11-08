#!/bin/bash
set -e
set -x
echo "Monolingual Evaluation"

for model in dev-gpt-35-turbo;
do
    for lang in en hi fr ko zh enhi
    do
        python -m mega.eval_xrisawoz \
            --root_dir "./xrisawoz_data/" \
            --num_learning_examples 4 \
            --seed 1618 \
            --model_name "$model" \
            --valid_fname "compressed_0.1_valid.json" \
            --language "$lang" \
            --substrate_llm
    done
done