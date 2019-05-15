import os
import glob
import h5py
import random
import numpy as np
from scipy.misc import bytescale

from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_transform



class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def read_hdf5(self, dom, idx, key='data'):
        path = self.paths[dom][idx]
        with h5py.File(path, 'r') as f_in:
            img = np.squeeze(np.array(f_in[key])).astype(np.float32)

        # Gray scale add channel axis
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = bytescale(img)  # Converts [0, 1] to [0, 255]
            img = np.concatenate((img, img, img), axis=2)

        img = self.transform(img)
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d, s in enumerate(self.sizes):
                    if index < s:
                        DA = d
                        break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path = self.read_hdf5(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, B_path = self.read_hdf5(DB, index_B)

            # Case where A_path == B_path
            while A_path == B_path:
                B_img, B_path = self.read_hdf5(DB, index_B)

            bundle.update({'B': B_img, 'DB': DB})

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'

