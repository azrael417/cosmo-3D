#!/bin/bash -l

#hdf file locking 
export HDF5_USE_FILE_LOCKING=FALSE

#mpi stuff
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"
export OMP_NUM_THREADS=1

for totalranks in 16; do

    # tag
    tag="orig_amp_threads_nranks${totalranks}"

    # files
    outfile="/data/profiles/timing_${tag}.out"
    #profilecmd="nsys profile --stats=true --mpi-impl=openmpi --trace=cuda,cublas,nvtx,osrt,mpi -f true -o /data/profiles/dali_pipe/profile_${tag}_%q{OMPI_COMM_WORLD_RANK}"

    #srun
    mpirun -np ${totalranks} ${mpioptions} \
	   ${profilecmd} $(which python) train_orig.py \
	   --yaml_config "config/UNet.yaml" \
	   --enable_amp \
	   --comm_mode "openmpi-nccl" --global_timing |& tee ${outfile}

done
