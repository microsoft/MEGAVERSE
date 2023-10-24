#!/bin/bash

for prompt_name in "Answer Given options"
    do
    for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf"
        do
        echo "ZS Cross Lingual Evaluation for ${model}"
        for lang in es eu hi id my ru sw te zh ar
        # for lang in te
        do
            k=2
            echo "Running for language $lang and prompt ${prompt_name} and k $k"
            python -m mega.hf_models.src.eval_xstory_cloze -d xstory_cloze -e melange -p en -t $lang --model "${model}" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb -k $k --timeout 30 --chat-prompt --use_api
        done
    done
done 

# echo "ZS Cross Lingual Evaluation for gpt4"
# for prompt_name in "Answer Given options"
# do
#     for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf"
#     do 
#         for lang in ar es eu hi id my ru sw te zh
#         # for lang in te
#         do
#             k=4
#             echo "Running for language $lang and prompt ${prompt_name} and k $k"
#             python -m mega.hf_models.src.eval_xstory_cloze -d xstory_cloze -e melange -p en -t $lang --model "${model}" --tgt_prompt_name "${prompt_name}" --temperature 0 --log_wandb -k $k --timeout 30
#         done
#     done
# done
