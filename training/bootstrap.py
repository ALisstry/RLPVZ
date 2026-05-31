import os

import torch


def configure_training_process() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    torch.set_num_threads(8)
