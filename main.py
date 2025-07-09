import os

import torch
from exp.exp_etr import ExpETR
from parameter_parser import parameter_parser
import warnings
import numpy as np
import random
from lib_utils.utils import get_parameter
warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parameter_parser()
    args = get_parameter(args)
    seed_everything(args['seed'])

    ExpETR(args)
