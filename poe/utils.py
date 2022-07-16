import logging
import os.path as osp
import random

import numpy as np
import wandb
from omegaconf import OmegaConf
from collections import defaultdict

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Statistics:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, batch_size=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, batch_size)

    def averages(self):
        """
        Compute averages from meters. Handle tensors vs floats (always return a
        float)
        Parameters
        ----------
        meters : Dict[str, util.AverageMeter]
            Dict of average meters, whose averages may be of type ``float`` or ``torch.Tensor``
        Returns
        -------
        metrics : Dict[str, float]
            Average value of each metric
        """
        metrics = {m: vs.avg for m, vs in self.meters.items()}
        metrics = {
            m: v if isinstance(v, float) else v.item() for m, v in metrics.items()
        }
        return metrics

    def __str__(self):
        meter_str = ", ".join(f"{k}={v}" for k, v in self.meters.items())
        return f"Statistics({meter_str})"


def global_setup(args):
    config = OmegaConf.to_container(args, resolve=False, throw_on_missing=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.wandb.log:
        wandb.init(
            project=str(args.wandb.project),
            group=str(args.wandb.group),
            name=str(args.wandb.name),
            config=config,
        )
    logger.info("args: %s", config)


def current_model(*, _root_):
    if _root_.backend == "openai":
        model = _root_.openai.engine
    elif _root_.backend == "cohere":
        model = _root_.cohere.model
    else:
        raise ValueError(f"Unknown backend {_root_.backend}")
    return f"{_root_.backend}-{model}"


def basename(f):
    return osp.splitext(osp.basename(f))[0]


OmegaConf.register_new_resolver("current_model", current_model)
OmegaConf.register_new_resolver("basename", basename)
