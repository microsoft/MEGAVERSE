#!/bin/bash

echo "Evaluating for IN22-Gen"
for lang in asm_Beng
do
    echo "Running for language $lang"
    python -m mega.eval_in22 -p en -t $lang  -k 4 --model "gpt-35-turbo" -e gpt4v2 --temperature 0 --num_evals_per_sec 2 --eval_on_val -d indicqa
done
