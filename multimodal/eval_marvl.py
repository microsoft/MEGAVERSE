import os
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai

from utils import (
    run_hf_model,
    run_gpt4v,
    run_gemini,
    LANG2CODE,
    dump_prediction,
    get_arg_parser,
    load_marvl,
    load_hf_model,
)

load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]
hf_token = os.environ["HF_TOKEN"]


def main(args):
    INSTRUCTION = "Is the following statement in {language} correct with respect to the left and right images? Return `TRUE` if it is true, else `FALSE`.\n\nSTATEMENT: {statement}"

    ds = load_marvl(args.ds_path, split=args.language)
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
            completed_ids = set([json.loads(line.strip())["q_idx"] for line in f])
    except FileNotFoundError:
        completed_ids = set()

    for i, line in tqdm(enumerate(ds), total=len(ds)):
        if i in completed_ids:
            continue

        left_img = line["left_img"]
        right_img = line["right_img"]
        combined_img = line["horizontally_stacked_img"]

        annotation = (
            line["hypothesis"].strip()
            if args.prompting_strategy == "translate-test"
            else line["hypo_en"].strip()
        )
        label = line["label"].strip()

        instruction = INSTRUCTION.format(
            language=LANG2CODE[args.language], statement=annotation
        )

        if is_hf_model:
            predicted_label = run_hf_model(
                model=model,
                processor=processor,
                image=combined_img,
                instruction=instruction,
                temperature=args.temperature,
                do_sample=(args.temperature > 0.0),
                max_new_tokens=2,
            )
        else:
            try:
                if args.model.startswith("gpt"):
                    predicted_label = run_gpt4v(
                        client=client,
                        images=[left_img, right_img],
                        model_name=args.model,
                        instruction=instruction,
                        temperature=args.temperature,
                        max_tokens=2,
                    )
                elif args.model.startswith("gemini"):
                    predicted_label = run_gemini(
                        instruction=instruction,
                        images=[left_img, right_img],
                        model=model,
                    )
            except Exception as e:
                print(f"Error: {e}, Skipping {i}")
                continue

        output = {
            "language": args.language,
            "prediction": predicted_label,
            "label": label,
        }

        dump_prediction(output, i, args.outfname)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
