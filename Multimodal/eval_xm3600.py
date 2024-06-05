import os
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai

from utils import (
    LANG2CODE,
    run_gpt4v,
    run_gemini,
    run_hf_model,
    get_arg_parser,
    dump_prediction,
    load_xm3600,
    load_hf_model,
)

load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]
hf_token = os.environ["HF_TOKEN"]


def main(args):
    INSTRUCTION = (
        "Generate a **brief** coco style caption for the given image in **{language}**"
    )

    ds = load_xm3600(args.ds_path, split=args.language)
    is_hf_model = False

    if args.model.startswith("gpt-4"):
        client = OpenAI(api_key=openai_api_key)
    elif args.model.startswith("gemini"):
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(args.model)
    else:
        model, processor = load_hf_model(args.model, hf_token=hf_token)
        is_hf_model = True

    try:
        with open(args.outfname, "r", encoding="utf-8") as f:
            completed_ids = set([json.loads(line)["q_idx"] for line in f])
    except FileNotFoundError:
        completed_ids = set()

    for i, line in tqdm(enumerate(ds), total=len(ds)):
        if i in completed_ids:
            continue

        image = line["image"]
        ref_captions = line["captions"]
        instruction = INSTRUCTION.format(language=LANG2CODE[args.language])

        if is_hf_model:
            prediction = run_hf_model(
                image=image,
                model=model,
                processor=processor,
                instruction=instruction,
                temperature=args.temperature,
                do_sample=(args.temperature > 0.0),
                max_new_tokens=args.max_new_tokens,
            )
        else:
            try:
                if args.model.startswith("gpt-4"):
                    prediction = run_gpt4v(
                        client=client,
                        images=[image],
                        model_name=args.model,
                        instruction=instruction,
                        temperature=args.temperature,
                        max_tokens=128,
                    )
                elif args.model.startswith("gemini"):
                    prediction = run_gemini(
                        model=model,
                        images=[image],
                        instruction=instruction,
                    )
            except Exception as e:
                print(f"Error: {e}, Skipping {i}")
                continue

        output = {
            "language": args.language,
            "captions": ref_captions,
            "prediction": prediction,
        }

        dump_prediction(output, i, args.outfname)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
