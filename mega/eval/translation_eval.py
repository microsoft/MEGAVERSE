import json
from IndicTransTokenizer import IndicEvaluator

ie = IndicEvaluator()


def evaluate(infname: str, outfname: str, tgt_lang: str) -> None:
    """
    Evaluate the translation predictions using the IndicTransTokenizer library.
    Args:
        infname (str): Path to the input file containing the predictions.
        outfname (str): Path to the output file where the evaluation results will be saved.
        tgt_lang (str): Target language code.
    """
    predictions = []
    references = []

    with open(infname, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line.strip())["prediction"]
            pred = data["prediction"].strip()
            ref = data["reference"].strip()
            if len(pred) > 0:
                predictions.append(pred)
                references.append(ref)

    score = ie.evaluate(tgt_lang=tgt_lang, preds=predictions, refs=references)

    with open(outfname, "w", encoding="utf-8") as f:
        f.write(json.dumps(score, ensure_ascii=False, indent=4))
