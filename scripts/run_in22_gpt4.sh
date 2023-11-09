#!/bin/bash

for dataset in IN22-Gen IN22-Conv;
do 
    echo "Evaluating for ${dataset}"
    for lang in asm_Beng ben_Beng guj_Gujr hin_Deva kas_Arab tel_Telu;
    do
        echo "Running for language ${lang}-eng_Latn"
        python -m mega.eval_in22 \
        -d $dataset \
        -s $lang \
        -t eng_Latn \
        -k 8 \
        -e gpt4v2 \
        --model gpt-4-32k \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 900 \
        --seed 42 \
        --test_examples 50

        echo "Running for language eng_Latn-${lang}"
        python -m mega.eval_in22 \
        -d $dataset \
        -s eng_Latn \
        -t $lang \
        -k 8 \
        -e gpt4v2 \
        --model gpt-4-32k \
        --temperature 0 \
        --num_evals_per_sec 2 \
        --save_dir results \
        --max_tokens 900 \
        --seed 42 \
        --test_examples 50
    done
done