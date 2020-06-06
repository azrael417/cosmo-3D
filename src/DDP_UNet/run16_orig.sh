#!/bin/bash -l

#number of ranks
totalranks=16

#hdf file locking 
export HDF5_USE_FILE_LOCKING=FALSE

#mpi stuff
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"

#srun
mpirun -np ${totalranks} ${mpioptions} \
     python train_orig.py \
     --yaml_config "config/UNet.yaml" \
     --comm_mode "openmpi-nccl"
