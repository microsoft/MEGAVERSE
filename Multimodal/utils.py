import json
import io
import torch
import backoff
from PIL import Image
from base64 import b64encode
from argparse import ArgumentParser
import google.ai.generativelanguage as glm
from datasets import Image, load_dataset
from transformers import AutoProcessor, AutoModelForPreTraining


device = "cuda" if torch.cuda.is_available() else "cpu"

LANG2CODE = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "zh": "Chinese",
    "id": "Indonesian",
    "tr": "Turkish",
    "sw": "Swahili",
    "ta": "Tamil",
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


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("-o", "--outfname", type=str, required=True)
    parser.add_argument(
        "-p",
        "--prompting_strategy",
        type=str,
        default="none",
        choices=["none", "translate-test"],
    )
    parser.add_argument("-l", "--language", type=str, default="en")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser


def dump_prediction(prediction, idx, output_file):
    prediction_ = {"q_idx": idx, "prediction": prediction}
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction_, ensure_ascii=False) + "\n")


def load_hf_model(model_id, attn_implementation="flash_attention_2", hf_token=None):
    model = AutoModelForPreTraining.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def image_to_bytes(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    return image_bytes.getvalue()


@backoff.on_exception(backoff.fibo, Exception, max_time=120)
def run_gpt4v(
    instruction,
    images,
    client,
    max_tokens=300,
    temperature=0.0,
    model_name="gpt-4-vision-preview",
):

    base64_images = [
        b64encode(image_to_bytes(image)).decode("utf-8") for image in images
    ]
    image_urls = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        for base64_image in base64_images
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    *image_urls,
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


@backoff.on_exception(backoff.fibo, Exception, max_time=120)
def run_gemini(
    instruction,
    images,
    model,
):

    image_urls = [
        glm.Part(
            inline_data=glm.Blob(mime_type="image/jpeg", data=image_to_bytes(image))
        )
        for image in images
    ]

    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(
                    text=instruction,
                ),
                *image_urls,
            ],
        ),
        safety_settings=GEMINI_SAFETY_SETTINGS,
    )

    return response.text.strip()


def run_hf_model(instruction, image, model, processor, **kwargs):
    prompt = f"USER: <image>\n{instruction}\nASSISTANT:"
    inputs = processor(prompt, image, return_tensors="pt").to(device, torch.bfloat16)
    outputs = model.generate(
        **inputs,
        max_new_tokens=kwargs.get("max_new_tokens", 256),
        do_sample=kwargs.get("do_sample", False),
        temperature=kwargs.get("temperature", 0.0),
    )
    return processor.decode(
        outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def load_marvl(hf_path="floschne/marvl", split="en"):
    ds = load_dataset(hf_path, split=split)
    current_columns = list(ds.features)

    ds = ds.map(
        lambda sample: {
            "resized_left_img_t": Image().decode_example(sample["resized_left_img"]),
            "resized_right_img_t": Image().decode_example(sample["resized_right_img"]),
            "horizontally_stacked_img_t": Image().decode_example(
                sample["horizontally_stacked_img"]
            ),
        },
        num_proc=32,
        remove_columns=current_columns,
    ).rename_columns(
        {
            "resized_left_img_t": "resized_left_img",
            "resized_right_img_t": "resized_right_img",
            "horizontally_stacked_img_t": "horizontally_stacked_img",
        }
    )

    return ds


def load_xm3600(hf_path="floschne/xm3600", split="en"):
    ds = load_dataset(hf_path, split=split)
    current_columns = list(ds.features)

    ds = ds.map(
        lambda sample: {
            "image_t": Image().decode_example(sample["image"]),
        },
        num_proc=32,
        remove_columns=current_columns,
    ).rename_columns({"image_t": "image"})
    return ds


# in case you want to define your own dataset, try to create it similar to floschne/xm3600 or floschne/marvl

# Define your own metrics and evaluation below, ex. Accuracy, ChrF, BLEU, CIDER, etc.
