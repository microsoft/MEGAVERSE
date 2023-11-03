import os
import openai
from dotenv import load_dotenv

load_dotenv("envs/melange.env")


def load_openai_env_variables():
    openai.api_base = os.environ["OPENAI_END_POINT"]
    openai.api_type = os.environ["OPENAI_API_TYPE"]
    openai.api_version = os.environ["OPENAI_API_VERSION"]
    openai.api_key = os.environ["OPENAI_API_KEY"]


HF_API_URL = os.environ["HF_API_URL"]
BLOOMZ_API_URL = os.environ["BLOOMZ_API_URL"]
OPEN_AI_KEY = os.environ["OPENAI_API_KEY"]
HF_API_KEY = os.environ["HF_API_KEY"]
BING_TRANSLATE_KEY = os.environ["BING_TRANSLATE_KEY"]
BING_TRANSLATE_ENDPOINT = os.environ["BING_TRANSLATE_ENDPOINT"]
COGNITIVE_API_ENDPOINT = os.environ["COGNITIVE_API_ENDPOINT"]
COGNITIVE_API_REGION = os.environ["COGNITIVE_API_REGION"]
COGNITIVE_API_VERSION = os.environ["COGNITIVE_API_VERSION"]
COGNITIVE_API_KEY = os.environ["COGNITIVE_API_KEY"]
