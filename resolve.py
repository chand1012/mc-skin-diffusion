import platform

import torch

def resolve_device() -> str:
    p = platform.system()
    cuda_is_available = torch.cuda.is_available()
    if p == "Darwin":
        return "mps"
    elif p == "Linux":
        if cuda_is_available:
            return "cuda"
        else:
            return "cpu"
    else:
        return "cuda" if cuda_is_available else "cpu"
