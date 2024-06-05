# Contamination analysis

## This is WIP

This code is implementation of the work done by Golchin et. al [here](https://arxiv.org/pdf/2311.06233.pdf).

### Generating Quiz inputs

1. Add your dataset specific template in `contamination/templates.py` under `TEMPLATES` dictionary. Also, add your verbalizers (if any), there.
2. Create a Pydantic model (you can follow the example of XNLI under `contamination/pydantic_models.py`). Make sure that keep your dataset_name constant over `TEMPLATES`, `PYDANTIC_REGISTRY`, `QUIZ_GENERATION_PROMPT_REGISTRY` and `GENERATED_RESPONSE_REGISTRY` dictionaries.
3. Create an appropriate yaml config and place it under `contamination_configs`. For ref, refer to `xnli_gpt_args.yaml` config
4. Run `python -m contamination.create_quiz_data contamination_configs/quiz_generation_configs/{your_config_name}`. It should start generating prompts and checkpoint at every instance.

### Running analysis
1. After your quizzes are generated for a specific dataset, you will now need to create the quiz prompt and run the quiz for the specific LLM.
2. The first step involves generating the prompt. Since every dataset has its specific quirk (Template), we will need to handle those quirks accordingly. To parse the generated quiz options, write a generated_response parser in `contamination/parse_generated_response_utils.py`. You can refer to the one written for `xnli`. Register the parser under `contamination/registry/generated_response_registry.py`.
3. Create your config file under `contamination_configs/quiz_answer_configs` and run it using `python -m contamination.run_analysis confimantion_config/quiz_answer_configs/{your_config_name}`. It should start generating answers and checkpointing those answers at every instance.


### Caveats
For now, this only supports huggingface's datasets. So, ensure that you copy the dataset_id from huggingface.

### Code directory structure

```
├── README.md
├── contamination_configs
│   ├── quiz_answer_configs
│   │   ├── xnli_gpt_args.yaml
│   │   └── xnli_palm_args.yaml
│   └── quiz_generation_configs
│       ├── xcopa_gpt_args.yaml.yml
│       ├── xcopa_palm_args.yaml
│       ├── xnli_gpt_args.yaml
│       └── xnli_palm_args.yaml
├── create_quiz_data.py
├── generated_quiz_answers
├── generated_quizzes
├── pydantic_models.py
├── registry
│   ├── langs_registry.py
│   ├── prompting_registry.py
│   └── pydantic_registry.py
├── run_analysis.py
├── templates.py
└── utils
    ├── general_utils.py
    ├── parse_generated_response_utils.py
    └── prompting_utils.py
```