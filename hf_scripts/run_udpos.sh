#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

MODEL_LIST="google/gemma-7b-it google/gemma-2b-it"
# for lang in pt ru
lang_codes=('af' 'ar' 'bg' 'de' 'el' 'en' 'es' 'et' 'eu' 'fa' 'fi' 'fr' 'he' 'hi' 'hu' 'id' 'it' 'ja' 'kk' 'ko' 'lt' 'mr' 'nl' 'pl' 'pt' 'ro' 'ru' 'ta' 'te' 'th' 'tl' 'tr' 'uk' 'ur' 'vi' 'wo' 'yo' 'zh')
for model in $MODEL_LIST
do
# for lang in en fr te bg de fi it pl nl
for lang in lang_codes
# for lang in ro te zh ar el et fi hi it lt pl ru tr vi bg eu fr hu ja mr pt ta uk wo
do
    echo "Running for lang $lang"
    python -m mega.eval_tag -d udpos -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat_wth_instruct --tgt_prompt_name structure_prompting_chat_wth_instruct -k 8 --model $model --max_tokens 100 --temperature 0 --eval_on_val --num_evals_per_sec 2  -e gpt4v2 --save_dir $save_dir --from_hf_hub --chat_prompt
    done
    # python -m mega.eval_tag -d udpos -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat --tgt_prompt_name structure_prompting_chat -k 8 --max_tokens 100 --temperature 0 --num_evals_per_sec 2 --model palm-32k -e gpt4v2
done
