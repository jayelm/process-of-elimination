"""
Active learning with OpenAI API
"""

import logging
import random
import time
from collections import Counter

import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf
from scipy.special import logsumexp
from tqdm import tqdm

from . import backends, utils, data

logger = logging.getLogger(__name__)


def eval_dataset(dataset, backend, args):
    stats = utils.Statistics()

    breakpoint()
    for ex in tqdm(
        dataset,
        desc=f"eval (prompt: {args.num_prompt_examples}, {args.prompt_format})",
    ):
        res = backend.generate(ex.text)
        pred_answer = res[-1]
        stats.update(
            acc=pred_answer == ex.answer,
            valid=pred_answer in data.LETTERS,
        )
        print(stats.averages())

    return stats.averages()


@hydra.main(config_path="conf", config_name="config")
def main(args):
    utils.global_setup(args)
    backend = backends.create(args)

    if args.dataset == "ecqa":
        dataset = data.ecqa.build_dataset(
            "./datasets/ECQA-Dataset/",
            prompt_format=args.prompt_format,
            num_prompt_examples=args.num_prompt_examples,
            instruction=args.instruction,
        )
    else:
        raise NotImplementedError(args.dataset)

    metrics = eval_dataset(dataset, backend, args)

    breakpoint()


if __name__ == "__main__":
    main()
