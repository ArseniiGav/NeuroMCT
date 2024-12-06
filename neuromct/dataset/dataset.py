import torch
from torch.utils.data import Dataset


class JMLDataset(Dataset):
    def __init__(
            self,
            dataset_type: str,
            path_to_processed_data: str,
            transform=None
        ):
        """
        Args:
            dataset_type (str): Type of the dataset ('training', 'val1', 'val2_1', 'val2_2', or 'val2_3').
            path_to_processed_data (str): Path to the directory containing the processed data files.
            transform (callable, optional): A function/transform to apply to the spectra.
        """
        if dataset_type in ["training", "val1", "val2_1", "val2_2", "val2_3"]:
            spectra_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_spectra.pt"
            params_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_params.pt"
            source_types_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_source_types.pt"
        else:
            raise ValueError("Invalid dataset_type! Choose between 'training', 'val1', 'val2_1', 'val2_2', and 'val2_3'.")

        self.spectra = torch.load(spectra_path, weights_only=True)
        self.params = torch.load(params_path, weights_only=True)
        self.source_types = torch.load(source_types_path, weights_only=True)
        self.transform = transform

        # Verify data consistency
        if len(self.spectra) != len(self.params) or len(self.spectra) != len(self.source_types):
            raise ValueError("Mismatch in the lengths of spectra, params, and source_types tensors.")

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectra = self.spectra[idx]
        params = self.params[idx]
        source_types = self.source_types[idx]
        sample = (spectra, params, source_types)

        if self.transform:
            sample = self.transform(sample)

        return sample
