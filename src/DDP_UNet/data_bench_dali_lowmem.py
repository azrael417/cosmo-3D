from utils.YParams import YParams
import os
import shutil as shu
import torch
from utils.data_loader_dali_lowmem import get_data_loader_distributed
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

# load parameters
params = YParams("config/UNet_transpose.yaml", "default")
device = torch.device("cuda:{}".format(comm_local_rank))

# setup
dist.barrier()
tstart = time.time()
# stage in?
if stage:
    # copy the input file into local DRAM for each socket:
    tmpfs_root = '/dev/shm'
    #tmpfs_root = '/tmp'
    #tmpfs_root = '/run/cosmo_data'
    gpus_per_socket = torch.cuda.device_count() // 2
    socket = 0 if comm_rank < gpus_per_socket else 1
    new_data_path = os.path.join(tmpfs_root, 'numa{}'.format(socket), os.path.basename(params.data_path))
    if comm_rank % (torch.cuda.device_count() / 2) == 0:
        if not os.path.isdir(os.path.dirname(new_data_path)):
            os.makedirs(os.path.dirname(new_data_path))
        if not os.path.isfile(new_data_path):
            shu.copyfile(params.data_path, new_data_path)

    # we need to wait till the stuff is copied
    dist.barrier()
    
    # change parameter path to new file
    params.data_path = new_data_path

# get data loader
train_data_loader = get_data_loader_distributed(params, comm_rank, comm_local_rank)
dist.barrier()
tend = time.time()
if comm_rank == 0:
    print("Setup: took {}s".format(tend - tstart))

# warmup
if comm_rank == 0:
    print("starting warmup")
for warm in range(num_warmup):
    for inp, tar in train_data_loader:
        pass

# timing
if comm_rank == 0:
    print("starting timing")

for run in range(num_benchmark):
    it = 0
    dist.barrier()
    tstart = time.time()
    for inp, tar in train_data_loader:
        it += 1
    dist.barrier()
    tend = time.time()
    
    if comm_rank == 0:
        print("Run {}: iterations took {}s for {} iterations ({} iter/s)".format(run, tend - tstart, it, float(it)/(tend - tstart)))

if comm_rank == 0:
    print("finished timing")
