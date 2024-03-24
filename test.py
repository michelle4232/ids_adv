import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
import torch as th
from torch import nn
from torch.autograd import Variable as V

import pandas as pd

df = pd.read_csv('datasets/target_model/CICIDS2017/target_model_CICIDS2017data.csv')
df