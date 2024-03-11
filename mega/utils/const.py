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
    "gemini-pro",
]

MODEL_TYPES = ["completion", "seq2seq"]

CHAT_MODELS = [
    "gpt-35-turbo",
    "gpt4_deployment",
    "gpt-4",
    "gpt-4-32k",
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

GEMINI_SUPPORTED_LANGUAGES_MAP = {
	"arabic": "ar",
	"bengali": "bn",
	"bulgarian": "bg",
	"chinese": "zh",
    "chinese_simplified": "zh-CN",
    "chinese_traditional": "zh-TW",
	"croatian": "hr",
	"czech": "cs",
	"danish": "da",
	"dutch": "nl",
	"english": "en",
	"estonian": "et",
	"finnish": "fi",
	"french": "fr",
	"german": "de",
	"greek": "el",
	"hebrew": "iw",
	"hindi": "hi",
	"hungarian": "hu",
	"indonesian": "id",
	"italian": "it",
	"japanese": "ja",
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
	"ukrainian": "uk",
	"vietnamese": "vi",
}

GEMINI_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]