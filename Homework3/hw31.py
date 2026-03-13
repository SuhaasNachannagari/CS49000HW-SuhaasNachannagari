import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import torch
import stanza
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


print("cuda available:", torch.cuda.is_available())
print("mps available:", torch.backends.mps.is_available())
print("device:", DEVICE)