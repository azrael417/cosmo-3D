#!/bin/bash

#nvidia-docker build -t tkurth/pytorch-bias_gan:latest .
cd ..

# Base
#nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo.deps \
#	      --build-arg PYVER=3.7 \
#	      --build-arg PYV=37 \
#	      -f docker/Dockerfile.deps.ubuntu .
#exit

## pytorch
#nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo.pytorch \
#	      --build-arg PYVER=3.7 \
#	      --build-arg PYV=37 \
#	      -f docker/Dockerfile.pytorch .
#exit

# dali
nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo.dali \
	      --build-arg PYVER=3.7 \
	      --build-arg PYV=37 \
	      -f docker/Dockerfile.dali .

# cosmo 3d
nvidia-docker build -t registry.services.nersc.gov/tkurth/pytorch-cosmo:latest -f docker/Dockerfile .
docker push registry.services.nersc.gov/tkurth/pytorch-cosmo:latest

#run docker test
#docker run --device=/dev/nvidia-fs0 --workdir "/opt/pytorch/numpy_reader/scripts" -it tkurth/pytorch-numpy_reader:latest ./reader_test.sh
#nvidia-docker run --workdir "/opt/pytorch/numpy_reader/scripts" -it tkurth/pytorch-numpy_reader:latest ./reader_test.sh
