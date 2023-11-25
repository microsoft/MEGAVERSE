import pandas as pd
import os
import pandas as pd


def parse_string(input_string):
    premises = []
    questions = []
    labels = []

    lines = input_string.split("\n")
    for line in lines:
        premise = ""
        question = ""
        label = ""

        if line.startswith("Premise:"):
            premise = line.replace("Premise:", "").strip()
            premises.append(premise)
        elif line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
            questions.append(question)
        elif line.startswith("Label:"):
            label = line.replace("Label:", "").strip()
            labels.append(label)

    return premises, questions, labels


def parse_df(df):
    sanitize_df = pd.DataFrame()
    for i, row in df.iterrows():
        premise, questions, labels = parse_string(row["generated_response"])
        if len(premise) != 3 or len(questions) != 3 or len(labels) != 3:
            print("skipping")
            continue
        sanitized_string = ""
        for premise, question, label in zip(premise, questions, labels):
            sanitized_string += f"Premise: {premise}\n"
            sanitized_string += f"Question: {question}\n"
            sanitized_string += f"Label: {label}\n"
            sanitized_string += "\n"

        # use concat instead of append below
        sanitize_df = pd.concat(
            [
                sanitize_df,
                pd.DataFrame(
                    {
                        "generated_response": [sanitized_string.strip()],
                        "original_example": [row["original_example"]],
                    }
                ),
            ]
        )

    return sanitize_df


def sanitize_csv_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "quiz_options.csv":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                sanitized_df = parse_df(df)
                sanitized_df.to_csv(file_path, index=False)
