import sys
import os
import time
import numpy as np
import argparse

import torch
#import torchvision
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from apex import amp, optimizers
#from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader import get_data_loader_distributed
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



def train(params, args, world_rank):
  # set device
  device = torch.device("cuda:{}".format(args.local_rank))
  
  logging.info('rank %d, begin data loader init'%world_rank)
  train_data_loader = get_data_loader_distributed(params, world_rank)
  logging.info('rank %d, data loader initialized'%world_rank)
  model = UNet.UNet(params).to(device)
  if not args.resuming:
    model.apply(model.get_weights_function(params.weight_init))

  optimizer = optimizers.FusedAdam(model.parameters(), lr = params.lr)
  #model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # for automatic mixed precision
  if params.distributed:
    #model = DDP(model)
    model = DDP(model, device_ids = [device.index], output_device = device.index)
    

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
      start = time.time()
      tr_time = 0.
      log_time = 0.

      epoch_step = 0
      for i, data in enumerate(train_data_loader, 0):
        torch.cuda.nvtx.range_push("cosmo3D:forward step {}".format(iters))
        tr_start = time.time()
        adjust_LR(optimizer, params, iters)
        inp, tar = map(lambda x: x.to(device), data)
        b_size = inp.size(0)
      
        model.zero_grad()
        gen = model(inp)
        loss = UNet.loss_func(gen, tar, params)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("cosmo3D:backward step {}".format(iters))
        loss.backward() # fixed precision
        optimizer.step()
        torch.cuda.nvtx.range_pop()

        iters += 1
        epoch_step += 1

        # ste pdone
        tr_end = time.time()
        tr_time += tr_end - tr_start


      # Output training stats
      log_start = time.time()
      if (params.validate == 1) and (world_rank == 0):
        gens = []
        tars = []
        with torch.no_grad():
          for i, data in enumerate(train_data_loader, 0):
            if i>=16:
              break
            inp, tar = map(lambda x: x.to(device), data)
            gen = model(inp)
            gens.append(gen.detach().cpu().numpy())
            tars.append(tar.detach().cpu().numpy())
            gens = np.concatenate(gens, axis=0)
            tars = np.concatenate(tars, axis=0)

        # Scalars
        #args.tboard_writer.add_scalar('G_loss', loss.item(), iters)

        # Plots
        fig, chi, L1score = meanL1(gens, tars)
        #args.tboard_writer.add_figure('pixhist', fig, iters, close=True)
        #args.tboard_writer.add_scalar('Metrics/chi', chi, iters)
        #args.tboard_writer.add_scalar('Metrics/rhoL1', L1score[0], iters)
        #args.tboard_writer.add_scalar('Metrics/vxL1', L1score[1], iters)
        #args.tboard_writer.add_scalar('Metrics/vyL1', L1score[2], iters)
        #args.tboard_writer.add_scalar('Metrics/vzL1', L1score[3], iters)
        #args.tboard_writer.add_scalar('Metrics/TL1', L1score[4], iters)
      
        fig = generate_images(inp.detach().cpu().numpy()[0], gens[-1], tars[-1])
        #args.tboard_writer.add_figure('genimg', fig, iters, close=True)
        
      log_end = time.time()
      log_time += log_end - log_start

      # Save checkpoint
      #torch.save({'iters': iters, 'epoch':epoch, 'model_state': model.state_dict(), 
      #            'optimizer_state_dict': optimizer.state_dict()}, params.checkpoint_path)
    
      end = time.time()
      if world_rank==0:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, end-start))
        logging.info('train step time = {} ({} steps), logging time = {}'.format(tr_time, epoch_step, log_time))
        logging.info('train samples/sec = {}'.format(epoch_step / tr_time))



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", default=0, type=int)
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
  parser.add_argument("--comm_mode", default='slurm-nccl', type=str)
  parser.add_argument("--config", default='default', type=str)
  
  args = parser.parse_args()
  
  run_num = args.run_num

  params = YParams(os.path.abspath(args.yaml_config), args.config)


  # get env variables
  params.distributed = True
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
  args.local_rank = comm_local_rank
  comm_port = "29500"
  os.environ["MASTER_ADDR"] = comm_addr
  os.environ["MASTER_PORT"] = comm_port

  # init process group
  dist.init_process_group(backend = "nccl",
                        rank = comm_rank,
                        world_size = comm_size)

  # set device here to avoid unnecessary surprises
  torch.cuda.set_device(comm_local_rank)

  torch.backends.cudnn.benchmark = True

  args.resuming = False

  # Set up directory
  baseDir = './expts/'
  expDir = os.path.join(baseDir, args.config+'/'+str(run_num)+'/')
  if  comm_rank == 0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir, exist_ok = True)
      os.makedirs(expDir+'training_checkpoints/', exist_ok = True)
  
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    params.log()
    #args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))

  params.experiment_dir = os.path.abspath(expDir)
  params.checkpoint_path = os.path.join(params.experiment_dir, 'training_checkpoints/ckpt.tar')
  if os.path.isfile(params.checkpoint_path):
    args.resuming=True

  train(params, args, comm_rank)
  #if world_rank == 0:
  #  args.tboard_writer.flush()
  #  args.tboard_writer.close()
  logging.info('DONE ---- rank %d'%comm_rank)

