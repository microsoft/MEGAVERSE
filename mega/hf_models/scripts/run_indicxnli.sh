#!/bin/bash
echo "Monolingual Evaluation"
for prompt_name in "GPT-3 style"
do
    for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf"
        do
            # for lang in ar bg de el es fr hi ru sw th tr ur vi zh
            for lang in as bn gu hi kn ml mr or pa ta te
            # for lang in gu
            do
                echo "Running for language $lang and prompt ${prompt_name}"
                python -m mega.hf_models.src.eval_xnli -p $lang -t $lang --pivot_prompt_name "${prompt_name}" --tgt_prompt_name "${prompt_name}" -k 4 --model "${model}" -e "melange" --chat-prompt --temperature 0 --log_wandb
            done
        done
done
