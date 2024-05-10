echo "Monolingual Evaluation"
for k in 8
do
    for lang in ar de el en es hi ro ru th tr vi zh
    do
        echo "Running for lang $lang and k $k"
        python -m mega.eval_qa_gptturbo -p en -t $lang -d xquad --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" --eval_on_val -k $k --model gemini-pro
    done
done