#!/bin/bash

#some parameters
#data_root=/home/cuda/tkurth/cam5_data

#inference runs
nvidia-docker run \
	      --ipc host \
              --device /dev/nvidia-fs0  \
	      --device /dev/nvidia-fs1  \
	      --device /dev/nvidia-fs2  \
	      --device /dev/nvidia-fs3  \
	      --device /dev/nvidia-fs4  \
	      --device /dev/nvidia-fs5  \
	      --device /dev/nvidia-fs6  \
	      --device /dev/nvidia-fs7  \
	      --device /dev/nvidia-fs8  \
	      --device /dev/nvidia-fs9  \
	      --device /dev/nvidia-fs10 \
	      --device /dev/nvidia-fs11 \
	      --device /dev/nvidia-fs12 \
	      --device /dev/nvidia-fs13 \
	      --device /dev/nvidia-fs14 \
	      --device /dev/nvidia-fs15 \
	      --volume "/raid1/data/tkurth:/data1:rw" \
	      --volume "/raid3/data/tkurth:/data3:rw" \
	      --volume "/raid5/data/tkurth:/data5:rw" \
	      --volume "/raid7/data/tkurth:/data7:rw" \
	      --workdir "/opt/deepCam" -it registry.services.nersc.gov/tkurth/pytorch-bias_gan:latest /bin/bash
