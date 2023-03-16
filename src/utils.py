import os
import random
from argparse import ArgumentParser

import numpy as np
import torch


def _set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _os_settings(args: ArgumentParser):
    os.environ["RANK"] = str(args.rank)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["MASTER_ADDR"] = str(args.host)
    os.environ["MASTER_PORT"] = str(args.port)


def setup_os_parameter_and_seed(args: ArgumentParser, seed: int):
    _set_seed(seed)
    _os_settings(args)
