SUPPORTED_MODELS = [
    "BLOOM",
    "BLOOMZ",
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "gpt-4-32k",
    "gpt-4",
    "dev-gpt-35-turbo",
    "dev-moonshot",
    "dev-ppo",
    "prod-ppo",
    "palm",
    "palm-32k",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "palm",
    "palm-32k",
    "google/gemma-7b-it",
    "google/gemma-7b",
    "google/gemma-2b-it",
    "google/gemma-2b"
]

MODEL_TYPES = ["completion", "seq2seq"]

CHAT_MODELS = [
    "gpt-35-turbo",
    "gpt4_deployment",
    "gpt-4",
    "gpt-4-32k",
    # "google/gemma-7b-it",
    # "google/gemma-2b-it"
    # "meta-llama/Llama-2-70b-chat-hf",
]

PALM_SUPPORTED_LANGUAGES_MAP = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "bengali": "bn",
    "bulgarian": "bg",
    "croation": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "herbrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "italian": "it",
    "japanese": "ja",
    "indonesian": "id",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "ukranian": "uk",
    "vietnamese": "vi",
}

PALM_MAPPING = {"palm": "text-bison@001", "palm-32k": "text-bison-32k"}
