import sys
import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from apex import optimizers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
#from utils.data_loader_dali_lowmem import get_data_loader_distributed
from utils.data_loader_dali import get_data_loader_distributed
#from utils.data_loader_dali_dummy import get_data_loader_distributed
from utils.plotting import generate_images, meanL1
from networks import UNet

def adjust_LR(optimizer, params, iternum):
  """Piecewise constant rate decay"""
  if params.distributed and iternum<5000:
    lr = params.ngpu*params.lr*(iternum/5000.) #warmup for distributed training
  elif iternum<40000:
    lr = params.ngpu*params.lr
  elif iternum>80000:
    lr = params.ngpu*params.lr/4.
  else:
    lr = params.ngpu*params.lr/2.
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr



def train(params, args, world_rank, local_rank):
  
  #logging info
  logging.info('rank {:d}, begin data loader init (local rank {:d})'.format(world_rank, local_rank))
  #train_data_loader = get_data_loader_distributed(params, world_rank)
  train_data_loader = get_data_loader_distributed(params, world_rank, local_rank)
  logging.info('rank %d, data loader initialized'%world_rank)

  # set device
  device = torch.device("cuda:{}".format(local_rank))
  
  model = UNet.UNet(params)
  model.to(device)
  if not args.resuming:
    model.apply(model.get_weights_function(params.weight_init))

  optimizer = optimizers.FusedAdam(model.parameters(), lr = params.lr)
  #model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # for automatic mixed precision
  if params.distributed:
    model = DDP(model, device_ids=[local_rank])

  # amp stuff
  gscaler = amp.GradScaler()
    
  iters = 0
  startEpoch = 0
  checkpoint = None
  if args.resuming:
    if world_rank==0:
      logging.info("Loading checkpoint %s"%params.checkpoint_path)
    checkpoint = torch.load(params.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    iters = checkpoint['iters']
    startEpoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  if world_rank==0:
    logging.info(model)
    logging.info("Starting Training Loop...")

    
  for epoch in range(startEpoch, startEpoch+params.num_epochs):
    start = time.time()
    nsteps = 0
    fw_time = 0.
    bw_time = 0.
    log_time = 0.

    model.train()
    step_time = time.time()
    for i, data in enumerate(train_data_loader, 0):
      iters += 1

      #adjust_LR(optimizer, params, iters)
      inp, tar = data

      if not args.io_only:

        # fw pass
        fw_time -= time.time()
        optimizer.zero_grad()
        with amp.autocast():
          gen = model(inp)
          loss = UNet.loss_func(gen, tar, params)
        fw_time += time.time()

        # bw pass
        bw_time -= time.time()
        gscaler.scale(loss).backward()
        gscaler.step(optimizer)
        gscaler.update()
        bw_time += time.time()
      
      nsteps += 1

    # epoch done
    dist.barrier()
    step_time = (time.time() - step_time) / float(nsteps)
    fw_time /= float(nsteps)
    bw_time /= float(nsteps)
    io_time = max([step_time - fw_time - bw_time, 0])
    iters_per_sec = 1. / step_time
    
    ## Output training stats
    #model.eval()
    #if world_rank==0:
    #  log_start = time.time()
    #  gens = []
    #  tars = []
    #  with torch.no_grad():
    #    for i, data in enumerate(train_data_loader, 0):
    #      if i>=16:
    #        break
    #      #inp, tar = map(lambda x: x.to(device), data)
    #      inp, tar = data
    #      gen = model(inp)
    #      gens.append(gen.detach().cpu().numpy())
    #      tars.append(tar.detach().cpu().numpy())
    #  gens = np.concatenate(gens, axis=0)
    #  tars = np.concatenate(tars, axis=0)
    #
    #  # Scalars
    #  args.tboard_writer.add_scalar('G_loss', loss.item(), iters)
    #
    #  # Plots
    #  fig, chi, L1score = meanL1(gens, tars)
    #  args.tboard_writer.add_figure('pixhist', fig, iters, close=True)
    #  args.tboard_writer.add_scalar('Metrics/chi', chi, iters)
    #  args.tboard_writer.add_scalar('Metrics/rhoL1', L1score[0], iters)
    #  args.tboard_writer.add_scalar('Metrics/vxL1', L1score[1], iters)
    #  args.tboard_writer.add_scalar('Metrics/vyL1', L1score[2], iters)
    #  args.tboard_writer.add_scalar('Metrics/vzL1', L1score[3], iters)
    #  args.tboard_writer.add_scalar('Metrics/TL1', L1score[4], iters)
    #  
    #  fig = generate_images(inp.detach().cpu().numpy()[0], gens[-1], tars[-1])
    #  args.tboard_writer.add_figure('genimg', fig, iters, close=True)
    #  log_end = time.time()
    #  log_time += log_end - log_start

    #  # Save checkpoint
    #  torch.save({'iters': iters, 'epoch':epoch, 'model_state': model.state_dict(), 
    #              'optimizer_state_dict': optimizer.state_dict()}, params.checkpoint_path)
    
    end = time.time()
    if world_rank==0:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, end-start))
        logging.info('total time / step = {}, fw time / step = {}, bw time / step = {}, exposed io time / step = {}, iters/s = {}, logging time = {}'
                     .format(step_time, fw_time, bw_time, io_time, iters_per_sec, log_time))

    # finalize
    dist.barrier()



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  parser.add_argument("--comm_mode", default='slurm-nccl', type=str)
  parser.add_argument("--io_only", action="store_true")
  args = parser.parse_args()
  
  run_num = args.run_num

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  params.distributed = True

  # get env variables
  if (args.comm_mode == "openmpi-nccl"):
    #use pmix server address: only works for single node
    addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
    comm_addr = addrport.split(":")[0]
    comm_rank = int(os.getenv('OMPI_COMM_WORLD_RANK',0))
    comm_size = int(os.getenv("OMPI_COMM_WORLD_SIZE",0))
  elif (args.comm_mode == "slurm-nccl"):
    comm_addr=os.getenv("SLURM_SRUN_COMM_HOST")
    comm_size = int(os.getenv("SLURM_NTASKS"))
    comm_rank = int(os.getenv("PMI_RANK"))
  
  # common stuff
  comm_local_rank = comm_rank % torch.cuda.device_count()
  comm_port = "29500"
  os.environ["MASTER_ADDR"] = comm_addr
  os.environ["MASTER_PORT"] = comm_port

  # init process group
  dist.init_process_group(backend = "nccl",
                        rank = comm_rank,
                        world_size = comm_size)

  #torch.backends.cudnn.benchmark = True
  args.resuming = False

  # set number of gpu
  params.ngpu = comm_size
  
  # Set up directory
  baseDir = './expts/'
  expDir = os.path.join(baseDir, args.config+'/'+str(run_num)+'/')
  if  comm_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir, exist_ok=True)
      os.makedirs(expDir+'training_checkpoints/', exist_ok=True)
  
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    params.log()
    #args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))

  params.experiment_dir = os.path.abspath(expDir)
  params.checkpoint_path = os.path.join(params.experiment_dir, 'training_checkpoints/ckpt.tar')
  if os.path.isfile(params.checkpoint_path):
    args.resuming=True

  train(params, args, comm_rank, comm_local_rank)
  #if comm_rank == 0:
  #  args.tboard_writer.flush()
  #  args.tboard_writer.close()
  logging.info('DONE ---- rank %d'%comm_rank)

