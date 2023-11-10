#!/bin/bash

echo "Evaluating for XQUAD"
# for lang in ar de es hi vi zh en
for lang in en ar de el es hi ro ru th tr vi zh
do
    echo "Running for language $lang"
    python -m mega.eval_qa_gptturbo -p en -t $lang --pivot_prompt_name "answer_given_context_and_question" \
     --tgt_prompt_name "answer_given_context_and_question" -k 8 --model dev-moonshot -e gpt4v2 \
      --temperature 0 --num_evals_per_sec 2 -d xquad --eval_on_val --substrate_prompt --chat_prompt
done