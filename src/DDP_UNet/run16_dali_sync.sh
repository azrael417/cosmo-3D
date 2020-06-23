#!/bin/bash -l

#hdf file locking 
export HDF5_USE_FILE_LOCKING=FALSE

#mpi stuff
mpioptions="--allow-run-as-root --map-by ppr:8:socket:PE=3"


for totalranks in 1 2 4 8 16; do
    # tag
    tag="dali-sync_no-amp_threads_nranks${totalranks}"

    # files
    outfile="/data/profiles/timing_${tag}.out"
    #profilecmd="nsys profile --stats=true --mpi-impl=openmpi --trace=cuda,cublas,nvtx,osrt,mpi -f true -o /data/profiles/dali_pipe/profile_${tag}_%q{OMPI_COMM_WORLD_RANK}"

    #srun
    mpirun -np ${totalranks} ${mpioptions} \
	   ${profilecmd} $(which python) train_dali_sync.py \
	   --yaml_config "config/UNet_transpose.yaml" \
	   --comm_mode "openmpi-nccl" |& tee ${outfile}

done
