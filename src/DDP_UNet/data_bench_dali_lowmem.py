from utils.YParams import YParams
import torch
from utils.data_loader_dali_lowmem import get_data_loader_distributed
import time

#load parameters
params = YParams("config/UNet_transpose.yaml", "default")
device = torch.device("cuda:0")

# get data loader
train_data_loader = get_data_loader_distributed(params, 0)

it = 0
tstart = time.time()
for inp, tar in train_data_loader:
    it += 1
tend = time.time()
print("Iterations took {}s for {} iterations ({} iter/s)".format(tend - tstart, it, float(it)/(tend - tstart)))
