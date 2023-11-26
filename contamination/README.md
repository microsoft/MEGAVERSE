# Contamination analysis

## This is WIP

This repo is implementation of the work done by Golchin et. al [here](https://arxiv.org/pdf/2311.06233.pdf).

### Generating Quiz inputs

1. Add your dataset specific template in `contamination/templates.py` under `TEMPLATES` dictionary. Also, add your verbalizers (if any), there.
2. Create a Pydantic model (you can follow the example of XNLI under `contamination/pydantic_models.py`). Make sure that keep your dataset_name constant over `TEMPLATES`, `PYDANTIC_REGISTRY`, `QUIZ_GENERATION_PROMPT_REGISTRY` and `GENERATED_RESPONSE_REGISTRY` dictionaries.
3. Create an appropriate yaml config and place it under `contamination_configs`. For ref, refer to `xnli_gpt_args.yaml` config
4. Run `python -m contimantion.create_quiz_data contamination_configs/quiz_generation_configs/{your_config_name}`. It should start generating prompts and checkpoint at every instance.

### Running analysis
1. After your quizzes are generated for a specific dataset, you will now need to create the quiz prompt and run the quiz for the specific LLM.


### Caveats
For now, this only supports huggingface's datasets. So, ensure that you copy the dataset_id from huggingface.
