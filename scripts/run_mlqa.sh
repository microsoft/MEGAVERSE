echo "Monolingual Evaluation"
for k in 8
do
    for lang in en ar de es hi vi zh
    do
        echo "Running for lang $lang and k $k"
        python -m mega.eval_qa_gptturbo -p $lang -t $lang -d mlqa --pivot_prompt_name "answer_given_context_and_question" --tgt_prompt_name "answer_given_context_and_question" --eval_on_val -k $k --model gemini-pro
    done
done