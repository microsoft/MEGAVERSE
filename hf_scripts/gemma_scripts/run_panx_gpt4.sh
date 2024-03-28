#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

MODEL_LIST="google/gemma-7b-it google/gemma-2b-it"
echo $save_dir
# for model in "google/gemma-7b-it" "google/gemma-2b-it"
for model in $MODEL_LIST
# echo $model
do
    echo "Mono lingual eval for $model"
for lang in en ja fi es af ar az bg bn de el et eu fa fr gu he hi hu id it jv ka kk ko lt ml ru vi my th tr yo zh
do
    python -m mega.eval_tag -d panx -p $lang -t $lang  --pivot_prompt_name structure_prompting_chat_wth_instruct --tgt_prompt_name structure_prompting_chat_wth_instruct -k 8 --model $model --max_tokens 100 --temperature 0 --eval_on_val --num_evals_per_sec 2  -e gpt4v2 --save_dir $save_dir --from_hf_hub --chat_prompt
done
done



# for model in "google/gemma-7b-it" 
# do 
#     echo "Mono lingual eval for $model"
#     for prompt_name in "Answer Given options"
#     do
#         # for lang in ar en es eu hi id my ru sw te zh
#         for lang in ar id my ru sw te
#         # for lang in te
#         do
#             k=8
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.eval_xstory_cloze -d xstory_cloze -e gpt4v2 -p $lang -t $lang --model $model --tgt_prompt_name "${prompt_name}" --temperature 0 -k $k --timeout 30 --chat_prompt --from_hf_hub --save_dir $save_dir
#         done
#     done
# done



# hi ja