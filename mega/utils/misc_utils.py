import json


def dump_predictions(idx, pred, label, response_logger_file):
    obj = {"q_idx": idx, "prediction": pred, "label": label}
    with open(response_logger_file, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.replace('</s>', '')
    s = s.strip()
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))