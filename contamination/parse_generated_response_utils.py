from typing import List, Dict


def generate_xnli_str_from_generated_response(
    generated_response: List[Dict[str, str]]
) -> str:
    """Generate a string from the generated response.

    Args:
        generated_response (List[Dict[str,str]]): Generated response

    Returns:
        str: Generated string
    """
    generated_str = ""
    for i, option in enumerate(generated_response):
        generated_str += chr(ord("A") + i)
        generated_str += ") "
        generated_str += "Premise: " + option["Premise"].strip() + "\n"
        generated_str += "Question: " + option["Question"].strip() + "\n"
        generated_str += "Label: " + option["Label"].strip() + "\n"
        generated_str += "\n"
    return generated_str
