"""
Active learning with OpenAI API
"""

import logging
import random
import time
from collections import Counter

import datasets
import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf
from scipy.special import logsumexp
from tqdm import tqdm
from transformers import GPT2Tokenizer

from . import backends, utils, data

logger = logging.getLogger(__name__)


datasets.disable_caching()


class Dataset:
    def __init__(self, args):
        self.args = args
        self.data = datasets.load_dataset("csv", data_files=args.data.train_file)[
            "train"
        ]["text"]
        total_examples = args.al.seed_size + args.al.pool_size + args.data.test_size
        if len(self.data) < total_examples:
            logger.warning(
                f"Need {total_examples} examples but dataset only has {len(self.data)}"
            )

        self.data = self.data[:total_examples]
        self.prompt_prefix = self.args.data.prompt_prefix
        self.prompt_examples = None
        self.splits = None
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def reset(self):
        random.shuffle(self.data)
        (prompt_examples, pool_examples, test_examples,) = (
            self.data[: self.args.al.seed_size],
            self.data[
                self.args.al.seed_size : self.args.al.seed_size + self.args.al.pool_size
            ],
            self.data[self.args.al.seed_size + self.args.al.pool_size :],
        )
        assert len(test_examples) == self.args.data.test_size
        self.prompt_examples = prompt_examples
        self.splits = {
            "pool": pool_examples,
            "test": test_examples,
        }

    def add_examples_to_prompt(self, ex_ids):
        for ex_id in ex_ids:
            ex = self.splits["pool"].pop(ex_id)
            self.prompt_examples.append(ex)

    def examples(self, split="test"):
        prompt = "\n\n".join(self.prompt_examples)
        for example in self.splits[split]:
            processed = self.split_example(example)
            processed["prompt"] = f"{prompt}\n\n{processed['prompt']}"
            if self.prompt_prefix is not None:
                processed["prompt"] = f"{self.prompt_prefix}\n\n{processed['prompt']}"
            yield processed

    def split_example(self, example):
        i = example.index("Target:") + 8
        example_head, example_scratch = example[:i], example[i:]
        scratch_len = len(self.tokenizer(example_scratch)["input_ids"])
        example_target = (
            example.split("Target:")[-1].split("</scratch>")[-1][1:-2].strip()
        )
        return {
            "prompt": example_head,
            "target": example_target,
            "scratch": example_scratch,
            "scratch_len": scratch_len,
        }

    @property
    def prompt(self):
        p = "\n\n".join(self.prompt_examples)
        if self.prompt_prefix is not None:
            p = f"{self.prompt_prefix}\n\n{p}"
        return p

    @property
    def num_prompt_examples(self):
        return len(self.prompt_examples)


def acquire(dataset, backend, args):
    acq_scores = Counter()
    pool_examples = list(dataset.examples("pool"))
    if args.al.acq_size > len(pool_examples):
        raise RuntimeError(
            f"Acquiring {args.al.acq_size} examples but only {len(pool_examples)} in pool"
        )

    for i, ex in tqdm(
        enumerate(pool_examples),
        desc=f"acq (prompt {dataset.num_prompt_examples}, pool {len(pool_examples)}, choose {args.al.acq_size})",
        total=len(pool_examples),
    ):
        if args.al.method == "random":
            acq_score = np.random.random()
        elif args.al.method == "mte":
            # Mean token entropy
            raise NotImplementedError
        elif args.al.method == "mtlc":
            # Mean token least confidence
            res = backend.generate(ex)
            logprobs = np.array(res["logprobs"]["token_logprobs"])
            # Mean in log space
            logprobs_mean = logsumexp(logprobs) - np.log(len(logprobs))
            # Select items with the lowest logprob, i.e. the highest negative logprob
            acq_score = -logprobs_mean
        acq_scores[i] = acq_score

    acq_scores = acq_scores.most_common()
    sel_scores = sel_scores = acq_scores[: args.al.acq_size]

    sel_ids = [x[0] for x in sel_scores]
    dataset.add_examples_to_prompt(sel_ids)

    acq_scores_np = np.array([x[1] for x in acq_scores])
    sel_scores_np = np.array([x[1] for x in sel_scores])

    return {
        "acq_score_mean": acq_scores_np.mean(),
        "acq_score_min": acq_scores_np.min(),
        "acq_score_max": acq_scores_np.max(),
        "acq_score_var": acq_scores_np.var(),
        "acq_score_sel_mean": sel_scores_np.mean(),
        "acq_score_sel_min": sel_scores_np.min(),
        "acq_score_sel_max": sel_scores_np.max(),
        "acq_score_sel_var": sel_scores_np.var(),
    }


def eval(dataset, backend, args):
    hits = 0
    for num_examples, ex in tqdm(
        enumerate(dataset.examples("test"), start=1),
        desc=f"eval (prompt {dataset.num_prompt_examples}, eval {args.data.test_size})",
        total=args.data.test_size,
    ):
        res = backend.generate(ex)
        if res["answer"] is not None and res["answer"] == ex["target"]:
            hits += 1

    return {
        "acc": hits / num_examples,
        "prompt_examples": dataset.num_prompt_examples,
        "prompt_nchar": len(ex["prompt"]),
    }


def add_prefix(d, prefix):
    return {f"{prefix}/{k}": v for k, v in d.items()}


@hydra.main(config_path="conf", config_name="config")
def main(args):
    utils.global_setup(args)
    backend = backends.create(args)

    if args.dataset == "ecqa":
        prompt_dataset, dataset = data.ecqa.ECQADataset.prompt_and_dataset(
            "./datasets/ECQA-Dataset/"
        )
    else:
        raise NotImplementedError(args.dataset)

    # Apply format
    prompt_formatter = data.prompts.PROMPT_FORMATS[args.format]
    prompt_dataset = prompt_formatter(prompt_dataset)


if __name__ == "__main__":
    main()
