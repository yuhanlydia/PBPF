"""Seed projector initialization as well as the upstream runner's data sampling."""
import random
import runpy
import sys
import numpy as np
import torch
seed=int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 2701
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
runpy.run_path('scripts/run_rbr_repair_gate.py',run_name='__main__')
