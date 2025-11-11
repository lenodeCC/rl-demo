import torch


def auto_select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# default_device = auto_select_device()
default_device = torch.device("cpu")
