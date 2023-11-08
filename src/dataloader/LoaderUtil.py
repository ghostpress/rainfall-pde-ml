import torch
import numpy as np


class LoaderUtil:

    def __init__(self, dataset, split, batchsize, history=1):
        self.dataset = dataset
        self.split = split
        self.history = history
        self.batchsize = batchsize

        dataset.train_val_test_split(self.split)

    def create_loader(self, variable, dtype=torch.FloatTensor, shuffle=True, use_wind=True):
        data = []

        pairs = self.dataset.get_pairs(variable, history=self.history, use_wind=use_wind)

        for i in range(len(pairs[0])):
            dat = self.dataset.load_data_from_files(pairs[0][i])
            data.append(dat)

        ends = self.dataset.load_data_from_files(pairs[1])
        final_data = torch.from_numpy(np.array(data)).type(dtype)
        final_ends = torch.from_numpy(np.array(ends)).type(dtype)

        if use_wind:
            wind = self.dataset.load_data_from_files(pairs[2])
            final_wind = torch.from_numpy(np.array(wind)).type(dtype)

            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(final_data, final_ends, final_wind),
                                                 batch_size=self.batchsize, shuffle=shuffle)
        else:
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(final_data, final_ends),
                                                 batch_size=self.batchsize, shuffle=shuffle)

        return loader
