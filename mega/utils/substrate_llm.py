from msal import PublicClientApplication, SerializableTokenCache
import json
import os
import atexit
import requests
import os
import pprint

from dotenv import load_dotenv

load_dotenv("envs/melange.env")

SUBSTRATE_API_KEY = os.environ["SUBSTRATE_API_KEY"]
SUBSTRATE_API_AUTHORITY = os.environ["SUBSTRATE_API_AUTHORITY"]


class LLMClient:
    _ENDPOINT = "https://httpqas26-frontend-qasazap-prod-dsm02p.qas.binginternal.com/completions"
    _SCOPES = [f"api://{SUBSTRATE_API_KEY}/access"]

    def __init__(self):
        self._cache = SerializableTokenCache()
        atexit.register(
            lambda: open(".llmapi.bin", "w").write(self._cache.serialize())
            if self._cache.has_state_changed
            else None
        )

        self._app = PublicClientApplication(
            SUBSTRATE_API_KEY,
            authority=SUBSTRATE_API_AUTHORITY,
            token_cache=self._cache,
        )
        if os.path.exists(".llmapi.bin"):
            self._cache.deserialize(open(".llmapi.bin", "r").read())

    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "X-ModelType": model_name,
        }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers)
        return response.json()

    def send_stream_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "X-ModelType": model_name,
        }

        body = str.encode(json.dumps(request))
        response = requests.post(
            LLMClient._ENDPOINT, data=body, headers=headers, stream=True
        )
        for line in response.iter_lines():
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
                if text == "[DONE]":
                    break
                else:
                    yield json.loads(text)

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)

        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=LLMClient._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4)
                )

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]


def create_request_data(
    prompt,
    max_tokens=10,
    temperature=0,
    top_p=1,
    n=1,
    stream=False,
    logprops=None,
    stop=None,
):
    
    request_data = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        # "logprobs": logprops,
        "stop": stop,
    }
    # print("request data",request_data)
    return request_data


def exec_llm(
    request_data, llm_client, model_name="text-davinci-003", return_resp=False
):
    response = llm_client.send_request(model_name, request_data)
    try:
        text_result = response["choices"][0]["text"]
        if return_resp:
            return response, text_result
        else:
            return text_result
    except KeyError as e:
        print(response, "check")
        # no error handling
        return response


template = """<|im_start|>system\n===\n# OVERALL INSTRUCTIONS\n===\nYou are an NLP assistant whose purpose is to solve reading comprehension problems. You will be provided questions on a set of passages and you will need to provide the answer as it appears in the passage. The answer should be in the same language as the question and the passage.\n\n<|im_end|>\n<|im_start|>user\n1962 ல் பத்மஸ்ரீ விருது வழங்கப்படுவதற்கு கால் நூற்றாண்டுக்கு முன்பே இந்திய அரசால் அன்னை தெரேசா அடையாளங்காணப்பட்டுள்ளார். 1972-ல், பன்னாட்டு புரிந்துணர்வுக்கான ஜவகர்லால் நேரு விருது, 1980-ல் இந்தியாவின் உயரிய குடிமக்கள் விருதான பாரத ரத்னா உட்பட இந்திய உயர்விருதுகளை அடுத்த பத்தாண்டுகளில் பெற்றார். அவரது அதிகாரபூர்வ வாழ்க்கைச்சரித்திரம், இந்திய ஆட்சிப் பணியாளரான நவீன் சாவ்லாவால் எழுதப்பட்டு 1992இல் வெளியிடப்பட்டது. அன்னை தெரசாவைப் பற்றிய எல்லா இந்தியாரும் உயர்வாகப் பார்க்கவில்லை. கல்கத்தாவில் பிறந்து லண்டனில் வாழ்ந்து கொண்டிருக்கும் அவரது விமர்சகரான அரூப் ச்சேட்டர்ஜி அவர் வாழ்ந்த காலத்தில் கல்கத்தாவின் முக்கிய அங்கமாக இருக்கவில்லையெனக் குறிப்பிட்டுள்ளார். அன்னை தெரேசா தனது சொந்த ஊரான கல்கத்தாவின் புகழைக் குலைத்து விட்டதாகக் அவர்  குறை கூறியுள்ளார். பாரதிய ஜனதா கட்சி கிறிஸ்துவ தலித்துக்கள் விஷயத்தில், அவரோடு மோதிய போதிலும், அவரது மரணத்தின் போது அவரைப் புகழ்ந்து, இறுதி சடங்கிற்குத் தனது பதிளாளை அனுப்பியது. ஆனால் விஸ்வ ஹிந்து பரிஷத்தோ, அரசு மரியாதையுடன் கூடிய இறுதிச்சடங்கினை செய்யும் அரசாங்கத்தின் முடிவுக்கு எதிர்ப்புத் தெரிவித்தது. அதன் நிர்வாகி கிரிராஜ் கிஷோர், "அவரது முதல் கடமை கிறிஸ்துவத்திற்கே இருந்தது" என்றுக் கூறினார். பொது நல சேவை தற்செயலானது. மேலும் அவர் கிறிஸ்துவர்களுக்கு சாதகமானவரென்றும், இறப்பின் வாயிலிலிருப்போருக்கு இரகசிய திருமுழுக்கை மேற்கொள்ளுபவரென்றும் குற்றஞ்சாட்டினார். ஆனால் ஃப்ரண்ட் லைன் பத்திரிகையளித்த முதல் பக்கமரியாதையில் இக்குற்றச்சாட்டுகளை அப்பட்டமான தவறாக நிராகரித்துள்ளது. அவரது சேவையைப் பற்றிய கல்கத்தாவாசிகளின் எண்ணத்தில், எந்தத் தாக்கத்தையும் இவை விளைவித்துவிடவில்லை என்றும் கூறியிருக்கிறது. இப்புகழ்மாலையை சூட்டிய ஆசிரியர் அவரது தன்னலமற்ற சேவை செய்யும் சக்தியையும், தைரியத்தையும் புகழ்ந்தபோதிலும், பொது கூட்டங்களில் அவர் கருக்கலைப்பை எதிர்ப்பதையும், அதை அரசியல் நோக்கமில்லாததாகக் காட்டிக்கொள்வதையும் குறை கூறியுள்ளார். அண்மையில், இந்திய நாளேடான தி டெலிக்ராப், அவர் வறியவர்களின் துன்பத்தைப் போக்க ஏதேனும் செய்தாரா அல்லது உணர்வுபூர்வமாக நெறிகளைத் தூண்டும் நோக்கத்தோடு, நோயாளிகளையும் இறப்போரையும் பராமரித்து வந்தாடு நின்று விட்டாரா என்பதைக் குறித்து விசாரிக்கும்படி உரோமைக்கு வேண்டுகோள் விடுக்கப்பட்டுள்ளது என்று கூறியுள்ளது. செப்டம்பர் 1997 ல் இறுதிச்சடங்கிற்கு முன்னதாக ஒரு வார காலம் அன்னை தெரேசாவின் உடல் கொல்கத்தாவின் புனித தோமையார் ஆலயத்தில் பொதுமக்கள் பார்வைக்கு வைக்கப்பட்டிருந்தது. அனைத்து மத ஏழைகளுக்கும் அவர் ஆற்றிய தொண்டுக்குப் பரிகாரமாக, இந்திய அரசின் அரசு மரியாதையுடன் கூடிய இறுதிச்சடங்கு செய்யப்பட்டது. தெற்காசிய மற்றும் கிழக்காசிய சேவைகளுக்காக 1962-ல், பன்னாட்டுப் புரிந்துணர்தலுக்கான பிலிப்பைன்ஸின் ரமன் மக்சேசே விருதைப் பெற்றார். அயல்நாடுகளில் தாழ்த்தப்பட்ட ஏழைகளின் மீதான கருணை நிறைந்த கவனத்தையும், அதற்காகவே அவர் வழிநடத்திச் செல்லும் புதிய சபையையும் இவ்விருதின் தீர்வுக்குழுமம் அங்கீகரிக்கிறது என்று விருதில் குறிப்பிடப்பட்டிருந்தது. 1970களின் தொடக்கத்திற்குள் அன்னை தெரேசா அனைத்துலகாலும் அறியப்பட்டார். 1969இன் ஆவணப்படமான மேல்கம் முக்கேரிட்ஜ்-ன், சம்திங்க் பியுடிபுல் பார் காட் -ம், அதே தலைப்புடைய அவரது புத்தகமும் அவரது புகழுக்கு வித்திட்டவைகளில் முக்கியமானவை ஆகும். முக்கேரிட்ஜ் அந்நேரத்தில் ஒரு ஆன்மீக பயணத்தில் ஆழ்ந்திருந்தார். அச்செய்திப்படத்தின் படப்பிடிப்பின் போது மோசமான ஒளியமைப்பு சூழலில், குறிப்பாக இறப்பின் வாயிலிலிருப்போருக்கான இல்லங்களில் எடுக்கப்பட்ட காட்சிகள் பயன்பாட்டுக்கு உகந்தவையாக இல்லையென அவர் முதலில் நினைத்தாலும், இந்தியாவிலிருந்து திரும்பிய பின்னர் அக்காட்சிதொகுப்பு மிக நல்ல ஒளியமைப்புடன் வந்திருந்தது. அன்னை தெரேசாவிடமிருந்தே வந்த தெய்வீக ஒளியர்ப்புதம் இது என முக்கேரிட்ஜ் பறைசாற்றினார். அப்படப்பிடிப்புக் குழுவின் மற்றவர்கள் அது புதுவித அதிநுண்ணிய வகை கோடாக் படச்சுருளால் ஏற்பட்ட விளைவு என்றெண்ணினர். முக்கேரிட்ஜ் பின்னர் கத்தோலிக்கராகச் சமயம் மாறினார். இவ்வேளையில் கத்தோலிக்கர் உலகம் முழுவதும் அன்னை தெரேசாவைப் வெளிப்படையாய் புகழ ஆரம்பித்தனர். 1971-ல் திருத்தந்தை ஆறாம் பவுல், அமைதிக்கான முதல் திருத்தந்தை இருபத்திமூன்றாம் யோவான் பரிசை, அவரின் ஏழை எளியோருக்கான சேவையையும் கிறிஸ்துவ நெறியின் பறைசாற்றலையும், அமைதிக்கான முயற்சியையும் பாராட்டி அவருக்கு அளித்தார். அதன் பிறகு பேசெம் இன் டெர்ரிஸ் விருதைப் பெற்றார். தான் மரித்தநாளிலிருந்து அன்னை தெரேசா புனிதத்துவத்தினை நோக்கி வேகமாக முன்னேறித் தற்பொழுது முக்தி பேறினை எட்டியிருக்கிறார். அன்னை தெரேசா அரசாங்கங்களாலும், மக்கள் அமைப்புகளாலும் பெருமைப்படுத்தப்பட்டிருக்கிறார். ஆஸ்திரேலிய சமுதாயத்திற்கு மட்டுமல்லாது ஒட்டுமொத்த மனித குலத்துக்கும் செய்த சேவைக்காக, 1982-ல் அவர் ஆர்டர் ஆஃப் ஆஸ்திரேலியாவின் கௌரவ தோழர் என்ற விருதைப் பெற்றார். இங்கிலாந்தும், அமெரிக்காவும் அடுத்தடுத்து விருதுகள் வழங்கின.\n    Q: அன்னை தெரசாவுக்கு எப்போது பத்மஸ்ரீ விருது வழங்கப்பட்டது?\n\n    Referring to the passage above, the correct answer to the given question is:\n\n<|im_end|>\n<|im_start|>assistant\n\n"""

if __name__ == "__main__":
    llm_client = LLMClient()

    request_data = create_request_data(
        prompt=template,
        max_tokens=10,
        temperature=0,
        top_p=1,
        n=1,
        stream=False,
        logprops=None,
        stop=None,
    )
    completion = exec_llm(request_data, llm_client, model_name="dev-gpt-35-turbo")

    print(f"{completion}")
