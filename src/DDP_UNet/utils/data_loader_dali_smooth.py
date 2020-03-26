import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import h5py

#dali stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops            
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))


def get_data_loader_distributed(params, world_rank):
    train_loader = RandomCropDataLoader(params, num_workers=params.num_data_workers, device_id=0)
    return train_loader


class DaliInputIterator(object):
    def __init__(self, params):
        with h5py.File(params.data_path, 'r', driver="core", backing_store=False) as f:
            self.Hydro = f['Hydro'][...]
            self.Nbody = f['Nbody'][...]
        self.length = self.Nbody.shape[1]
        self.batch_size = params.batch_size
        self.size = params.data_size
        # compute the extract size such that the cubic diagonal of size^3 fits into extracted cube
        self.extract_size = int(self.size * np.sqrt(3))
        self.extract_size += 0 if self.extract_size % 2 == 0 else 1
        self.Nsamples = params.Nsamples
        self.rng = np.random.RandomState(seed=12345)
        self.max_bytes = 5 * (self.size**3) * 4
        self.transposed = False if params.transposed_input==0 else True
        print("Transposed Input" if self.transposed else "Original Input")

    def __iter__(self):
        self.i = 0
        self.n = self.Nsamples
        return self

    def __next__(self):
        rand = self.rng.randint(low=0, high=(self.length-self.size), size=(3))
        x = rand[0]
        y = rand[1]
        z = rand[2]
        if self.transposed:
            inp = np.expand_dims(np.copy(self.Nbody[x:x+self.extract_size, y:y+self.extract_size, z:z+self.extract_size, :]), axis=0)
            tar = np.expand_dims(np.copy(self.Hydro[x:x+self.extract_size, y:y+self.extract_size, z:z+self.extract_size, :]), axis=0)
        else:
            inp = np.expand_dims(np.copy(self.Nbody[:, x:x+self.extract_size, y:y+self.extract_size, z:z+self.extract_size]), axis=0)
            tar = np.expand_dims(np.copy(self.Hydro[:, x:x+self.extract_size, y:y+self.extract_size, z:z+self.extract_size]), axis=0)

        ##rotation axis
        #angles = self.rng.random_sample(size=(1,2)).astype(np.float32)
        #angles[:, 0] *= 2. * np.pi
        #angles[:, 1] *= np.pi
        #axis = np.stack([ np.cos(angles[:, 0]) * np.sin(angles[:, 1]), np.sin(angles[:, 0]) * np.sin(angles[:, 1]), np.cos(angles[:, 1]) ], axis=1)
    
        return inp, tar
    
    next = __next__


class DaliPipeline(Pipeline):
    def __init__(self, params, num_threads, device_id):
        super(DaliPipeline, self).__init__(params.batch_size,
                                           num_threads,
                                           device_id,
                                           seed=12)
        dii = DaliInputIterator(params)
        self.iterator = iter(dii)
        self.input = ops.ExternalSource()
        self.target = ops.ExternalSource()
        self.do_rotate = True if params.rotate_input==1 else False
        print("Enable Rotation" if self.do_rotate else "Disable Rotation")
        self.rng_angle = ops.Uniform(device = "cpu",
                                     range = [0., 180.])
        self.rng_axis = ops.Uniform(device = "cpu",
                                    range = [-1., 1.],
                                    shape=(3))
        self.rotate = ops.Rotate(device = "gpu",
                                 axis = (1.,1.,1.),
                                 interp_type = types.INTERP_LINEAR,
                                 keep_size=True)
        self.transpose = ops.Transpose(device = "gpu",
                                       perm=[3,0,1,2])
        self.crop = ops.Crop(device = "gpu",
                             crop = (dii.size, dii.size, dii.size))

    def define_graph(self):
        self.inp = self.input()
        self.tar = self.target()
        if self.do_rotate:
            # rotate
            angle = self.rng_angle()
            #axis = self.rng_axis()
            dinp = self.rotate(self.inp.gpu(), angle=angle)
            dtar = self.rotate(self.tar.gpu(), angle=angle)
            # crop because rotation can be bigger
            dinp = self.crop(dinp)
            dtar = self.crop(dtar)
            # transpose data
            self.dinp = self.transpose(dinp)
            self.dtar = self.transpose(dtar)
        else:
            self.dinp = self.transpose(self.inp.gpu())
            self.dtar = self.transpose(self.tar.gpu())
        return self.dinp, self.dtar

    def iter_setup(self):
        inp, tar = self.iterator.next()
        self.feed_input(self.inp, inp)
        self.feed_input(self.tar, tar)
        #self.feed_input(self.axs, axs)


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

