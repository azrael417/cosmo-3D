from utils.YParams import YParams
import torch
from utils.data_loader_dali import get_data_loader_distributed
import time
import torch.distributed as dist

#init process group
dist.init_process_group(backend = "mpi")
comm_rank = dist.get_rank()
comm_local_rank = comm_rank % torch.cuda.device_count()

#load parameters
params = YParams("config/UNet_transpose.yaml", "default")
device = torch.device("cuda:{}".format(comm_local_rank))

# get data loader
train_data_loader = get_data_loader_distributed(params, comm_rank, comm_local_rank)

it = 0
tstart = time.time()
for inp, tar in train_data_loader:
    it += 1
tend = time.time()
print("Iterations took {}s for {} iterations ({} iter/s)".format(tend - tstart, it, float(it)/(tend - tstart)))
