"""ECQA dataset"""


import os.path as osp
import pandas as pd
import random


from .common import Example, Dataset
from typing import List


QUESTION_STRING = "{q_text}\nA: {q_op1}\nB: {q_op2}\nC: {q_op3}\nD: {q_op4}\nE: {q_op5}"


def _process_example(x):
    question = QUESTION_STRING.format(**x)

    choices = []
    for i in range(5):
        if i == x["q_ans_index"]:
            # Positive explanation can have several explanations. Just take
            # the first
            explanation = x[f"a_op{i + 1}"].split("\n\n\n")[0]
        else:
            explanation = x[f"a_op{i + 1}"]
        choices.append({
            "text": x[f"q_op{i + 1}"],
            "explanation": explanation,
        })

    return Example(
        question=question,
        choices=choices,
        answer="ABCDE"[x["q_ans_index"]],
        answer_index=x["q_ans_index"],
    )


def build_dataset(
    path: str,
    prompt_format: str = "direct",
    num_prompt_examples: int = 5,
    **dataset_kwargs,
):
    test_path = osp.join(path, "cqa_data_test.csv")
    if not osp.exists(test_path):
        raise RuntimeError(f"Couldn't find {test_path}, did you run datasets/prepare_ecqa.sh?")

    val_path = osp.join(path, "cqa_data_val.csv")
    if not osp.exists(val_path):
        raise RuntimeError(f"Couldn't find {val_path}, did you run datasets/prepare_ecqa.sh?")

    test_data = pd.read_csv(test_path)
    val_data = pd.read_csv(val_path)

    test_data = [_process_example(x) for _, x in test_data.iterrows()]
    val_data = [_process_example(x) for _, x in val_data.iterrows()]

    random.shuffle(test_data)
    random.shuffle(val_data)

    # Pick prompt from val data.
    prompt_data = val_data[:num_prompt_examples]

    prompt = Dataset(prompt_data, format=prompt_format).get_prompt()
    dataset = Dataset(test_data, format="test", prompt=prompt, **dataset_kwargs)

    return dataset
