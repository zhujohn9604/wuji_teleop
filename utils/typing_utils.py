import torch
import numpy as np
from typing import Dict, Union

Array = Union[np.ndarray, torch.Tensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
