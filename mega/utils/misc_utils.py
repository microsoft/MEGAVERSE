import json


def dump_predictions(idx, pred, label, response_logger_file):
    obj = {"q_idx": idx, "prediction": pred, "label": label}
    with open(response_logger_file, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
