from torch.utils.data import Dataset


class JMLDataset(Dataset):
    def __init__(self, spectra, params, source_type):
        assert len(data) == len(labels) == len(parameters), \
            "All tensors must have the same length"
        self.spectra = spectra
        self.params = params
        self.source_type = source_type

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.params[idx], self.source_type[idx]
