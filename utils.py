import re
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

def untuple(x):
    return x[0] if isinstance(x, tuple) else x
    
