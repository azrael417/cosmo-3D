import os
from utils.YParams import YParams
import torch
from utils.data_loader import get_data_loader_distributed
import time
import torch.distributed as dist

# global parameters
num_warmup = 1
num_benchmark = 5
stage = True

# get env variables
comm_addr=os.getenv("SLURM_SRUN_COMM_HOST")
comm_size = int(os.getenv("SLURM_NTASKS"))
comm_rank = int(os.getenv("PMI_RANK"))
comm_local_rank = comm_rank % torch.cuda.device_count()
comm_port = "29500"
os.environ["MASTER_ADDR"] = comm_addr
os.environ["MASTER_PORT"] = comm_port

# init process group
dist.init_process_group(backend = "nccl",
                        rank = comm_rank,
                        world_size = comm_size)

#load parameters
params = YParams("config/UNet.yaml", "default")
device = torch.device("cuda:{}".format(comm_local_rank))

# setup
dist.barrier()
tstart = time.time()
train_data_loader = get_data_loader_distributed(params, comm_size)
dist.barrier()
tend = time.time()
if comm_rank == 0:
    print("Setup: took {}s".format(tend - tstart))

# warmup
if comm_rank == 0:
    print("starting warmup")
for warm in range(num_warmup):
    for inp, tar in train_data_loader:
        inp = inp.to(device)
        tar = tar.to(device)

# timing
if comm_rank == 0:
    print("starting timing")

for run in range(num_benchmark):
    it = 0
    dist.barrier()
    tstart = time.time()
    for inp, tar in train_data_loader:
        inp = inp.to(device)
        tar = tar.to(device)
        it += 1
    dist.barrier()
    tend = time.time()

    if comm_rank == 0:
        print("Run {}: iterations took {}s for {} iterations ({} iter/s)".format(run, tend - tstart, it, float(it)/(tend - tstart)))

if comm_rank == 0:
    print("finished timing")

