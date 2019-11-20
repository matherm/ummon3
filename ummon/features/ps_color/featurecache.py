import torch
import numpy as np
from pathlib import Path
import os
import PIL.Image

__all__ = ['FeatureCache']

import hashlib


def props(cls):
    return [i for i in cls.__dict__.keys() if i[:1] != '_']


def hash_obj(obj):
    hashes = []
    for key in props(obj):
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        hash_val = hashlib.md5(str(getattr(obj, key)).encode('utf-8')).hexdigest()
        hashes.append(hash_key + hash_val)
    hash_output = hashlib.md5("".join(hashes).encode("utf-8")).hexdigest()
    return hash_output


def hash(obj):
    return hashlib.md5(obj).hexdigest()


class FeatureCache():
    """
    Simple feature cache.

    Usage
    ======
            transform = FeatureCache(VGG(features="pool4"), cachedir = None, clearcache = True)
            transform(tensor)

            OR

            vgg = FeatureCache(VGG19Features(features="pool4"), cachedir = None, clearcache = True)
            my_transforms = transforms.Compose([transforms.ToTensor(), vgg])
            test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms)

    Input
    ======
        *tensor (torch.Tensor) with shape (B x 3 x min 32 x 32)

    Return
    =====
        *feature (torch.Tensor)

    """

    def __init__(self, transform, cachedir="__cache__", clearcache=False, force_tensor_output=True):
        """
        Parameters
        ----------
            *transform (torchvison.transform) : the transformation to cache
            *cachedir (str) : an directory for caching computed matrices
            *clearcache (bool) : deletes cache on object construction

        """
        self.transform = transform
        self.cachedir = cachedir
        self.clearcache = clearcache
        self.force_tensor = force_tensor_output
        self.unique_transformation_hash = hash_obj(self.transform)

        self.cache_hits = 0
        self.cache_misses = 0

        # create cache dir
        if os.path.exists(self.cachedir) == False:
            os.makedirs(self.cachedir)

        if clearcache == True:
            [os.remove(os.path.join(self.cachedir, f)) for f in os.listdir(self.cachedir) if f.endswith(".npy")]

    def __call__(self, x):
        if isinstance(x, PIL.Image.Image):
            x = np.array(x)
        is_numpy = isinstance(x, np.ndarray)
        is_cuda = x.is_cuda if not is_numpy else False

        # Handle cache lookup
        if is_numpy:
            input_raw_data = x.data.tobytes()
        else:
            input_raw_data = x.detach().to('cpu').numpy().data.tobytes()
        fname = "".join(self.unique_transformation_hash) + "_" + str(hash(input_raw_data))
        # path = str(self.cachedir + "/ummon_" + fname + ".npy")
        path = os.path.join(self.cachedir, 'cache_{}.npy'.format(fname))

        if Path(path).exists():
            self.cache_hits += 1  # stats
            cached_raw_data = np.load(path)
            if is_numpy and not self.force_tensor:
                return cached_raw_data
            elif is_cuda:
                return torch.from_numpy(cached_raw_data).to('cuda')
            else:
                return torch.from_numpy(cached_raw_data)
        else:
            self.cache_misses += 1  # stats

        # Compute transformation
        y = self.transform(x)

        ## Handle cache put
        np.save(path, y.to('cpu').detach().numpy())

        return y