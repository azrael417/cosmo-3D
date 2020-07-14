import torch
import numpy as np
import cupy as cp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import h5py

#concurrent futures
import concurrent.futures as cf

#dali stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops            
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))


def get_data_loader_distributed(params, world_rank, device_id=0):
    train_loader = RandomCropDataLoader(params, num_workers=params.num_data_workers, device_id=device_id)
    return train_loader


class DaliParameterIterator(object):
    def __init__(self, params):
        self.rng = np.random.RandomState(seed=12345)
        self.Nsamples = params.Nsamples

    def __iter__(self):
        self.i = 0
        self.n = self.Nsamples
        return self

    def __next__(self):
        # select op
        index = int(self.rng.randint(low=0, high=24))

        # select angle and axis
        if index == 0:
            axis = [1, 0, 0]
            angle = 0
        elif index == 1:
            axis = [1, 0, 0]
            angle = 90
        elif index == 2:
            axis = [1, 0, 0]
            angle = -90
        elif index == 3:
            axis = [0, 1, 0]
            angle = 90
        elif index == 4:
            axis = [0, 1, 0]
            angle = -90
        elif index == 5:
            axis = [0, 0, 1]
            angle = 90
        elif index == 6:
            axis = [0, 0, 1]
            angle = -90

        elif index == 7:
            axis = [1, 0, 0]
            angle = 180
        elif index == 8:
            axis = [0, 1, 0]
            angle = 180
        elif index == 9:
            axis = [0, 0, 1]
            angle = 180

        elif index == 10:
            axis = [1, 1, 0]
            angle = 180
        elif index == 11:
            axis = [1, 0, 1]
            angle = 180
        elif index == 12:
            axis = [0, 1, 1]
            angle = 180
        elif index == 13:
            axis = [1, -1, 0]
            angle = 180
        elif index == 14:
            axis = [1, 0, -1]
            angle = 180
        elif index == 15:
            axis = [0, 1, -1]
            angle = 180

        elif index == 16:
            axis = [1, 1, 1]
            angle = 120
        elif index == 17:
            axis = [-1, 1, 1]
            angle = 120
        elif index == 18:
            axis = [1, -1, 1]
            angle = 120
        elif index == 19:
            axis = [1, 1, -1]
            angle = 120
        elif index == 20:
            axis = [-1, -1, 1]
            angle = 120
        elif index == 21:
            axis = [-1, 1, -1]
            angle = 120
        elif index == 22:
            axis = [1, -1, -1]
            angle = 120
        elif index == 23:
            axis = [-1, -1, -1]
            angle = 120

        # recast into appropriate shape
        axis = np.expand_dims(np.array(axis, dtype=np.float32), axis=0)
        angle = np.expand_dims(np.array((angle), dtype=np.float32), axis=0)
        
        return axis, angle
    

class DaliInputIterator(object):
    def pin(self, array):
        mem = cp.cuda.alloc_pinned_memory(array.nbytes)
        ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        ret[...] = array
        return ret
    
    def __init__(self, params, device_id):
        # set device
        self.device_id = device_id
        cp.cuda.Device(self.device_id).use()

        # memory pool
        self.pinned_memory_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self.pinned_memory_pool.malloc)
        
        # stage in data
        with h5py.File(params.data_path, 'r') as f:
            self.Hydro = f['Hydro'][...]
            self.Nbody = f['Nbody'][...]

        # other parameters
        self.length = self.Nbody.shape[1]
        self.size = params.data_size
        self.Nsamples = params.Nsamples
        self.rng = np.random.RandomState(seed=12345)
        self.max_bytes = 5 * (self.size**3) * 4
        self.transposed = False if params.transposed_input==0 else True
        print("Transposed Input" if self.transposed else "Original Input")
        # threadpool
        self.executor = cf.ThreadPoolExecutor(max_workers = 2)
        # prepared arrays for double buffering
        self.curr_buff = 0
        self.Nbody_buff = None
        self.Hydro_buff = None
        if self.transposed:
            # CPU
            self.Nbody_buff_cpu = self.pin(np.zeros((1, self.size, self.size, self.size, self.Nbody.shape[3]), dtype=self.Nbody.dtype))
            self.Hydro_buff_cpu = self.pin(np.zeros((1, self.size, self.size, self.size, self.Hydro.shape[3]), dtype=self.Hydro.dtype))
            
            # GPU
            self.Nbody_buff_gpu	= [cp.zeros((1, self.size, self.size, self.size, self.Nbody.shape[3]), dtype=self.Nbody.dtype),
                                   cp.zeros((1, self.size, self.size, self.size, self.Nbody.shape[3]), dtype=self.Nbody.dtype)]
            self.Hydro_buff_gpu = [cp.zeros((1, self.size, self.size, self.size, self.Hydro.shape[3]), dtype=self.Hydro.dtype),
                                   cp.zeros((1, self.size, self.size, self.size, self.Hydro.shape[3]), dtype=self.Hydro.dtype)]
            
        else:
            # CPU
            self.Nbody_buff_cpu = self.pin(np.zeros((1, self.Nbody.shape[0], self.size, self.size, self.size), dtype=self.Nbody.dtype))
            self.Hydro_buff_cpu = self.pin(np.zeros((1, self.Hydro.shape[0], self.size, self.size, self.size), dtype=self.Hydro.dtype))
            
            # GPU
            self.Nbody_buff_gpu = [cp.zeros((1, self.Nbody.shape[0], self.size, self.size, self.size), dtype=self.Nbody.dtype),
                                   cp.zeros((1, self.Nbody.shape[0], self.size, self.size, self.size), dtype=self.Nbody.dtype)]
            self.Hydro_buff_gpu = [cp.zeros((1, self.Hydro.shape[0], self.size, self.size, self.size), dtype=self.Hydro.dtype),
                                   cp.zeros((1, self.Hydro.shape[0], self.size, self.size, self.size), dtype=self.Hydro.dtype)]

        # submit data fetch
        buff_ind = self.curr_buff
        self.future = self.executor.submit(self.get_rand_slice, buff_ind)

    def __iter__(self):
        self.i = 0
        self.n = self.Nsamples
        return self
        
    
    def get_rand_slice(self, buff_id):
        # RNG
        rand = self.rng.randint(low=0, high=(self.length-self.size), size=(3))
        x = rand[0]
        y = rand[1]
        z = rand[2]
        
        # Slice
        if self.transposed:
            self.Nbody_buff_cpu[0, ...] = self.Nbody[x:x+self.size, y:y+self.size, z:z+self.size, :]
            self.Hydro_buff_cpu[0, ...] = self.Hydro[x:x+self.size, y:y+self.size, z:z+self.size, :]
        else:
            self.Nbody_buff_cpu[0, ...] = self.Nbody[:, x:x+self.size, y:y+self.size, z:z+self.size]
            self.Hydro_buff_cpu[0, ...] = self.Hydro[:, x:x+self.size, y:y+self.size, z:z+self.size]

        # upload
        axis = None,
        angle = None
        with cp.cuda.stream.Stream() as stream_htod:
            self.Nbody_buff_gpu[buff_id].set(self.Nbody_buff_cpu)
            self.Hydro_buff_gpu[buff_id].set(self.Hydro_buff_cpu)
            stream_htod.synchronize()
        
        # create handles
        inp = self.Nbody_buff_gpu[buff_id]
        tar = self.Hydro_buff_gpu[buff_id]
        
        return inp, tar
    
    def __next__(self):
        torch.cuda.nvtx.range_push("DaliInputIterator:next")
        # wait for batch load to complete
        inp, tar = self.future.result()

        # submit new work before proceeding
        self.curr_buff = (self.curr_buff + 1) % 2
        buff_ind = self.curr_buff
        self.future = self.executor.submit(self.get_rand_slice, buff_ind)
        torch.cuda.nvtx.range_pop()
        
        return inp, tar
    
    next = __next__


class DaliPipeline(Pipeline):
    def __init__(self, params, num_threads, device_id):
        super(DaliPipeline, self).__init__(params.batch_size,
                                           num_threads,
                                           device_id,
                                           seed=12)
        dii = DaliInputIterator(params, device_id)
        dpi = DaliParameterIterator(params)
        self.no_copy = params.no_copy
        if self.no_copy:
            print("Use Zero Copy ES")
        self.source = ops.ExternalSource(device = "gpu",
                                         source = dii,
                                         num_outputs = 2,
                                         layout = ["DHWC", "DHWC"],
                                         no_copy = self.no_copy)
        self.params = ops.ExternalSource(device = "cpu",
                                         source = dpi,
                                         num_outputs = 2)
        self.do_rotate = True if params.rotate_input==1 else False
        print("Enable Rotation" if self.do_rotate else "Disable Rotation")
        self.rotate = ops.Rotate(device = "gpu",
                                 interp_type = types.INTERP_LINEAR)
        self.transpose = ops.Transpose(device = "gpu",
                                       perm=[3,0,1,2])

    def define_graph(self):
        self.inp, self.tar = self.source()
        self.axis, self.angle = self.params()

        if self.do_rotate:
            # rotate
            self.dinp = self.rotate(self.inp, axis=self.axis, angle=self.angle)
            self.dtar = self.rotate(self.tar, axis=self.axis, angle=self.angle)

            # transpose
            self.dinp = self.transpose(self.dinp)
            self.dtar = self.transpose(self.dtar)
            
        else:
            self.dinp = self.transpose(self.inp)
            self.dtar = self.transpose(self.tar)
        
        return self.dinp, self.dtar


class RandomCropDataLoader(object):
    """Random crops"""
    def __init__(self, params, num_workers=1, device_id=0):
        self.pipe = DaliPipeline(params, num_threads=num_workers, device_id=device_id)
        self.pipe.build()
        self.length = params.Nsamples
        self.iterator = DALIGenericIterator([self.pipe], ['inp', 'tar'], self.length, auto_reset = True)
        
    def __len__(self):
        return self.length

    def __iter__(self):
        for token in self.iterator:
            inp = token[0]['inp']
            tar = token[0]['tar']
            yield inp, tar


class RandomRotator(object):
    """Composable transform that applies random 3D rotations by right angles.
       Adapted from tf code:
       https://github.com/doogesh/halo_painting/blob/master/wasserstein_halo_mapping_network.ipynb"""

    def __init__(self):
        self.rot = {1:  lambda x: x[:, ::-1, ::-1, :],
                    2:  lambda x: x[:, ::-1, :, ::-1],
                    3:  lambda x: x[:, :, ::-1, ::-1],
                    4:  lambda x: x.transpose([0, 2, 1, 3])[:, ::-1, :, :],
                    5:  lambda x: x.transpose([0, 2, 1, 3])[:, ::-1, :, ::-1],
                    6:  lambda x: x.transpose([0, 2, 1, 3])[:, :, ::-1, :],
                    7:  lambda x: x.transpose([0, 2, 1, 3])[:, :, ::-1, ::-1], 
                    8:  lambda x: x.transpose([0, 3, 2, 1])[:, ::-1, :, :],
                    9:  lambda x: x.transpose([0, 3, 2, 1])[:, ::-1, ::-1, :],
                    10: lambda x: x.transpose([0, 3, 2, 1])[:, :, :, ::-1],
                    11: lambda x: x.transpose([0, 3, 2, 1])[:, :, ::-1, ::-1],
                    12: lambda x: x.transpose([0, 1, 3, 2])[:, :, ::-1, :],
                    13: lambda x: x.transpose([0, 1, 3, 2])[:, ::-1, ::-1, :],
                    14: lambda x: x.transpose([0, 1, 3, 2])[:, :, :, ::-1],
                    15: lambda x: x.transpose([0, 1, 3, 2])[:, ::-1, :, ::-1],
                    16: lambda x: x.transpose([0, 2, 3, 1])[:, ::-1, ::-1, :],
                    17: lambda x: x.transpose([0, 2, 3, 1])[:, :, ::-1, ::-1],
                    18: lambda x: x.transpose([0, 2, 3, 1])[:, ::-1, :, ::-1],
                    19: lambda x: x.transpose([0, 2, 3, 1])[:, ::-1, ::-1, ::-1],
                    20: lambda x: x.transpose([0, 3, 1, 2])[:, ::-1, ::-1, :],
                    21: lambda x: x.transpose([0, 3, 1, 2])[:, ::-1, :, ::-1],
                    22: lambda x: x.transpose([0, 3, 1, 2])[:, :, ::-1, ::-1],
                    23: lambda x: x.transpose([0, 3, 1, 2])[:, ::-1, ::-1, ::-1],
                    24: lambda x: x}

    def __call__(self, x, rand):
        return self.rot[rand](x)




class RandomCropDataset_PTflip(Dataset):
    """Random crops, using pytorch logic for rotations"""
    def __init__(self, params, Nbody, Hydro):
        self.Hydro = Hydro
        self.Nbody = Nbody
        self.length = Nbody.shape[1]
        self.size = params.data_size
        self.Nsamples = 1000
        self.rotate = RandomRotator_PTflip()

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        x = np.random.randint(low=0, high=self.length-self.size)
        if self.length-self.size-self.size==0:
            y = 0
        else:
            y = np.random.randint(low=0, high=self.length-self.size-self.size)
        z = np.random.randint(low=0, high=self.length-self.size)
        inp = self.Nbody[:, x:x+self.size, y:y+self.size, z:z+self.size]
        tar = self.Hydro[:, x:x+self.size, y:y+self.size, z:z+self.size]
        rand = np.random.randint(low=1, high=25)
        return self.rotate(torch.as_tensor(inp), rand), self.rotate(torch.as_tensor(tar), rand)


def flip_along(x, dims):
    for dim in dims:
        x = flip_(x, dim)
    return x

def flip_(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1)
    return x[tuple(indices)]

class RandomRotator_PTflip(object):
    """Composable transform that applies random 3D rotations by right angles in PT.
       Adapted from tf code:
       https://github.com/doogesh/halo_painting/blob/master/wasserstein_halo_mapping_network.ipynb"""

    def __init__(self):
        # Using pytorch flip() fn
        self.rot = {1:  lambda x: x.flip([1,2]),
                    2:  lambda x: x.flip([1,3]),
                    3:  lambda x: x.flip([2,3]),
                    4:  lambda x: x.permute([0, 2, 1, 3]).flip([1]),
                    5:  lambda x: x.permute([0, 2, 1, 3]).flip([1,3]),
                    6:  lambda x: x.permute([0, 2, 1, 3]).flip([2]),
                    7:  lambda x: x.permute([0, 2, 1, 3]).flip([2,3]),
                    8:  lambda x: x.permute([0, 3, 2, 1]).flip([1]),
                    9:  lambda x: x.permute([0, 3, 2, 1]).flip([1,2]),
                    10: lambda x: x.permute([0, 3, 2, 1]).flip([3]),
                    11: lambda x: x.permute([0, 3, 2, 1]).flip([2,3]),
                    12: lambda x: x.permute([0, 1, 3, 2]).flip([2]),
                    13: lambda x: x.permute([0, 1, 3, 2]).flip([1,2]),
                    14: lambda x: x.permute([0, 1, 3, 2]).flip([3]),
                    15: lambda x: x.permute([0, 1, 3, 2]).flip([1,3]),
                    16: lambda x: x.permute([0, 2, 3, 1]).flip([1,2]),
                    17: lambda x: x.permute([0, 2, 3, 1]).flip([2,3]),
                    18: lambda x: x.permute([0, 2, 3, 1]).flip([1,3]),
                    19: lambda x: x.permute([0, 2, 3, 1]).flip([1,2,3]),
                    20: lambda x: x.permute([0, 3, 1, 2]).flip([1,2]),
                    21: lambda x: x.permute([0, 3, 1, 2]).flip([1,3]),
                    22: lambda x: x.permute([0, 3, 1, 2]).flip([2,3]),
                    23: lambda x: x.permute([0, 3, 1, 2]).flip([1,2,3]),
                    24: lambda x: x}
        '''
        # Using custom pytorch re-indexing
        self.rot = {1:  lambda x: flip_along(x,[1,2]),
                    2:  lambda x: flip_along(x,[1,3]),
                    3:  lambda x: flip_along(x,[2,3]),
                    4:  lambda x: flip_along(x.permute([0, 2, 1, 3]),[1]),
                    5:  lambda x: flip_along(x.permute([0, 2, 1, 3]),[1,3]),
                    6:  lambda x: flip_along(x.permute([0, 2, 1, 3]),[2]),
                    7:  lambda x: flip_along(x.permute([0, 2, 1, 3]),[2,3]),
                    8:  lambda x: flip_along(x.permute([0, 3, 2, 1]),[1]),
                    9:  lambda x: flip_along(x.permute([0, 3, 2, 1]),[1,2]),
                    10: lambda x: flip_along(x.permute([0, 3, 2, 1]),[3]),
                    11: lambda x: flip_along(x.permute([0, 3, 2, 1]),[2,3]),
                    12: lambda x: flip_along(x.permute([0, 1, 3, 2]),[2]),
                    13: lambda x: flip_along(x.permute([0, 1, 3, 2]),[1,2]),
                    14: lambda x: flip_along(x.permute([0, 1, 3, 2]),[3]),
                    15: lambda x: flip_along(x.permute([0, 1, 3, 2]),[1,3]),
                    16: lambda x: flip_along(x.permute([0, 2, 3, 1]),[1,2]),
                    17: lambda x: flip_along(x.permute([0, 2, 3, 1]),[2,3]),
                    18: lambda x: flip_along(x.permute([0, 2, 3, 1]),[1,3]),
                    19: lambda x: flip_along(x.permute([0, 2, 3, 1]),[1,2,3]),
                    20: lambda x: flip_along(x.permute([0, 3, 1, 2]),[1,2]),
                    21: lambda x: flip_along(x.permute([0, 3, 1, 2]),[1,3]),
                    22: lambda x: flip_along(x.permute([0, 3, 1, 2]),[2,3]),
                    23: lambda x: flip_along(x.permute([0, 3, 1, 2]),[1,2,3]),
                    24: lambda x: x}
        '''
    def __call__(self, x, rand):
        return self.rot[rand](x)

