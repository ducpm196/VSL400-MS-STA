import os
import random

import numpy as np
import torch


def enforce_reproducibility_seed(seed: int = 42) -> None:
    """Lock random sources for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[OK] Đã khóa cứng thực nghiệm hệ thống với Mã Seed cố định: {seed}")
