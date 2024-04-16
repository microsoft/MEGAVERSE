import os
import openai
from dotenv import load_dotenv

load_dotenv("envs/melange.env")


def load_openai_env_variables():
    # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=os.environ["OPENAI_END_POINT"])'
    openai.api_base = os.environ["OPENAI_END_POINT"]


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
