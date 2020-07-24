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
#from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
#import utils.data_loader_dali_cupy as dl
import utils.data_loader_dali_cupy_opt as dl 
from nvidia.dali.plugin.pytorch import DALIGenericIterator
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

  # set device
  device = torch.device("cuda:{}".format(local_rank))

  # data loader
  pipe = dl.DaliPipeline(params, num_threads=params.num_data_workers, device_id = device.index)
  pipe.build()
  train_data_loader = DALIGenericIterator([pipe], ['inp', 'tar'], params.Nsamples, auto_reset = True)
  logging.info('rank %d, data loader initialized'%world_rank)

  model = UNet.UNet(params).to(device)
  
  if not args.resuming:
    model.apply(model.get_weights_function(params.weight_init))

  optimizer = optimizers.FusedAdam(model.parameters(), lr = params.lr)
  #model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # for automatic mixed precision
  if params.distributed:
    model = DDP(model, device_ids = [device.index], output_device = device.index)

  # loss
  criterion = UNet.CosmoLoss(params.LAMBDA_2)
    
  # amp stuff
  if args.enable_amp:
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

  with torch.autograd.profiler.emit_nvtx():
    for epoch in range(startEpoch, startEpoch+params.num_epochs):
      
      if args.global_timing:
        dist.barrier()

      start = time.time()
      epoch_step = 0
      tr_time = 0.
      fw_time = 0.
      bw_time = 0.
      log_time = 0.

      model.train()
      for data in train_data_loader:
        torch.cuda.nvtx.range_push("cosmo3D:step {}".format(iters))
        tr_start = time.time()
        adjust_LR(optimizer, params, iters)

        # fetch data
        inp = data[0]["inp"]
        tar = data[0]["tar"]
      
        if not args.io_only:
          torch.cuda.nvtx.range_push("cosmo3D:forward {}".format(iters))
          # fw pass
          fw_time -= time.time()
          optimizer.zero_grad()
          with amp.autocast(args.enable_amp):
            gen = model(inp)
            loss = criterion(gen, tar)
          fw_time += time.time()
          torch.cuda.nvtx.range_pop()

          # bw pass
          torch.cuda.nvtx.range_push("cosmo3D:backward {}".format(iters))
          bw_time -= time.time()
          if args.enable_amp:
            gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()
          else:
            loss.backward()
            optimizer.step()
          bw_time += time.time()
          torch.cuda.nvtx.range_pop()

        iters += 1
        epoch_step += 1

        # step done
        tr_end = time.time()
        tr_time += tr_end - tr_start
        torch.cuda.nvtx.range_pop()

      # epoch done
      if args.global_timing:
        dist.barrier()
        
      end = time.time()
      epoch_time = end - start
      step_time = epoch_time / float(epoch_step)
      tr_time /= float(epoch_step)
      fw_time /= float(epoch_step)
      bw_time /= float(epoch_step)
      io_time = max([step_time - fw_time - bw_time, 0])
      iters_per_sec = 1. / step_time
      fw_per_sec = 1. / tr_time

      if world_rank==0:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, epoch_time))
        logging.info('train step time = {} ({} steps), logging time = {}'.format(tr_time, epoch_step, log_time))
        logging.info('train samples/sec = {} fw steps/sec = {}'.format(iters_per_sec, fw_per_sec))      


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  parser.add_argument("--comm_mode", default='slurm-nccl', type=str)
  parser.add_argument("--io_only", action="store_true")
  parser.add_argument("--enable_amp", action="store_true")
  parser.add_argument("--no_copy", action="store_true")
  parser.add_argument("--global_timing", action="store_true")
  args = parser.parse_args()
  
  run_num = args.run_num

  params = YParams(os.path.abspath(args.yaml_config), args.config)

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

  params.distributed = True if comm_size > 1 else False
  
  # init process group
  dist.init_process_group(backend = "nccl",
                        rank = comm_rank,
                        world_size = comm_size)

  # set device here to avoid unnecessary surprises
  torch.cuda.set_device(comm_local_rank)

  torch.backends.cudnn.benchmark = True
  args.resuming = False

  # ES stuff
  params.no_copy = args.no_copy
  
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

